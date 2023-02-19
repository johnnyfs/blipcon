import datetime, random, os

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image, ImageFilter

from diffusers.models.unet_2d_blocks import DownEncoderBlock2D
from diffusers import AutoencoderKL
from diffusers.models.unet_2d_blocks import Upsample2D

b = 8
ch = 512
s = 64
t = torch.randn(b, ch, s, s)
t = t.view(b, ch, s * s).transpose(1, 2)
print(t.shape)
batch_size, seq_len, dim = t.shape # b, 4096, 512
head_size = 1
t = t.reshape(batch_size, seq_len, head_size, dim // head_size) # b, 4096, 1, 512
t = t.permute(0, 2, 1, 3) # b, 1, 4096, 512
t = t.reshape(batch_size * head_size, seq_len, dim // head_size) # b*1, 4096, 512
print(t.shape)
u = torch.nn.Linear(ch, ch)(t)
v = torch.nn.Linear(ch, ch)(t)
w = torch.nn.Linear(ch, ch)(t)
attention_scores = torch.baddbmm(
    torch.empty(
        u.shape[0], # batch
        u.shape[1], # 4096
        u.shape[1], # 4096
        dtype=t.dtype,
        device=t.device,
    ),
    u, # shape is (batch, 4096, 512)
    v.transpose(-1, -2), # shape is (batch, 512, 4096)
    beta=0,
    alpha=1,
)
print(attention_scores.shape)
attention_probs = torch.softmax(attention_scores.float(), dim=-1).type(attention_scores.dtype) # b, 4096, 4096
print(attention_probs.shape)
print(f'{attention_probs.shape} x {w.shape} = {torch.bmm(attention_probs, w).shape}')
hidden_states = torch.bmm(attention_probs, w) # b, 4096, 512
print(hidden_states.shape)
hidden_states = hidden_states.transpose(1, 2).reshape(8, 512, 64, 64)
print(hidden_states.shape)
print(Upsample2D(512, use_conv=True, out_channels=512)(hidden_states).shape)
exit()


torch.cuda.empty_cache()

transform = transforms.Compose([
    transforms.ToTensor(),
])
untransform = transforms.Compose([
    transforms.ToPILImage(),
])

def quanitize(img):
    out = img.copy()
    for px in range(out.width):
        for py in range(out.height):
            c = img.getpixel((px, py))
            if c <= 64:
                out.putpixel((px, py), 0)
            elif c <= 128:
                out.putpixel((px, py), 84)
            elif c <= 192:
                out.putpixel((px, py), 170)
            else:
                out.putpixel((px, py), 255)
    return out

paths = [ 'input/dreamboy_village.bmp', 'input/zgb_uniques.png' ]
imgs = []
for path in paths:
    bank = Image.open(path).convert('L')
    # Copy each 16x16 tile from the bank to a new PIL image
    for y in range(0, bank.height, 16):
        for x in range(0, bank.width, 16):
            img = bank.crop((x, y, x + 16, y + 16))
            imgs.append(img)

ts = [ transform(img).to('cuda') for img in imgs ]

batch_size = 512
loader = torch.utils.data.DataLoader(
    dataset = ts,
    batch_size = batch_size,
    shuffle=True)

class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode = torch.nn.Sequential(
            #nn.Flatten(1),
            nn.Linear(16 * 16, 192),
            nn.ReLU(),
            nn.Linear(192, 64),
            nn.ReLU(),
            nn.Linear(64, 36),
            nn.ReLU(),
            nn.Linear(36, 18),
            nn.ReLU(),
            nn.Linear(18, 9),
        )
        self.decode = torch.nn.Sequential(
            nn.ReLU(),
            nn.Linear(9, 18),
            nn.ReLU(),
            nn.Linear(18, 36),
            nn.ReLU(),
            nn.Linear(36, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 16 * 16),
            nn.Sigmoid(),
            #nn.Unflatten(1, (1, 16, 16)),
        )
    
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(0.5)

        kernel_size = 3
        # Contracting path
        self.contract = nn.Conv2d(1, 4, kernel_size, padding='same')
        # Middle path
        self.middle = nn.Conv2d(4, 8, kernel_size, padding='same')
        # Expanding path
        self.up = nn.ConvTranspose2d(8, 4, kernel_size, 2, (1, 1), 1)
        # concat happens here
        self.expand1 = nn.Conv2d(8, 4, kernel_size, padding='same')
        self.expand2 = nn.Conv2d(4, 1, kernel_size, padding='same')

        self.do_drop = True

    def forward(self, x):
        # Contracting path
        x = self.relu(self.contract(x))
        c1_out = x
        x = self.pool(x)
        if self.do_drop:
            x = self.drop(x)

        # Middle path
        x = self.relu(self.middle(x))

        # Expanding path
        x = self.up(x)
        x = torch.cat((x, c1_out), 1)
        if self.do_drop:
            x = self.drop(x)
        x = self.relu(self.expand1(x))
        x = self.sigmoid(self.expand2(x))
        return x

net = Net()
net = net.to('cuda')
net.do_drop = True

# Validation using MSE Loss function
loss_function = torch.nn.MSELoss()
 
# Using an Adam Optimizer with lr = 0.1
optimizer = optim.Adam(
    net.parameters(),
    lr = 1e-1,
    weight_decay = 1e-8
)

epochs = 2000
outputs = []
losses = []
ts = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
note = 'net-fullish'
path = f'./output/{ts}-{note}'
os.mkdir(path)
noise_start = 200
for epoch in range(epochs):
    if epoch % 100 == 0:
        print(f'Epoch {epoch}')
    b = 0
    for batch in loader:
        old_batch = batch.clone().to('cuda')
        # Add noise
        if epoch > noise_start and epoch % 3 == 0:
            batch = transforms.Lambda(lambda x: x + torch.randn_like(x) * ((1.0 * epoch - noise_start) / epochs / 500))(batch)
        reconstructeds = net(batch)
        loss = loss_function(reconstructeds, old_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu())

        if epoch % 250 == 0 or epoch == epochs - 1:
            x, y = 0, 0
            new_bank = Image.new('L', (128, 16 * (batch_size // 8 + 1)))
            for i in range(len(reconstructeds)):
                r = reconstructeds[i].cpu()
                print(r)
                print(r.shape)
                print((r + r).shape)
                print((r + r))
                exit()
                img = untransform(r)
                new_bank.paste(img, (x, y))
                x += 16
                if x >= 128:
                    x = 0
                    y += 16
            #quanitize(new_bank)
            new_bank.save(f'{path}/{epoch}_{b}.png')
        b += 1


noise_img = Image.new('L', (16, 16))
for y in range(16):
    for x in range(16):
        noise_img.putpixel((x, y), random.choice([0, 84, 170, 255]))
noise_img.save(f'{path}/noise.png')
noise_img = transform(noise_img).to('cuda').unsqueeze(0)
for i in range(100):
    result = net(noise_img)
    if i % 10 == 0 or i == 99:
        result = quanitize(untransform(result.squeeze(1)))
        result.save(f'{path}/from_noise_{i}.png')
        noise_img = transform(result).to('cuda').unsqueeze(0)
    else:
        noise_img = result

plt.style.use('fivethirtyeight')
plt.xlabel('Iterations')
plt.ylabel('Loss')
 
# Plotting the last 100 values
losses = [ loss.detach().numpy() for loss in losses ]
loss_imgs = plt.plot(losses[-epochs:])
plt.savefig(f'{path}/loss.png')
        