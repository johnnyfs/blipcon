from enum import Enum
from typing import List, Optional

import datetime, random, os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image, ImageFilter

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

# Print iterations progress
def progress(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

class MiniDist:
    def __init__(self, params):
        self.params = params
        # Split the latent channels into mean and log variance
        mean, logvar = torch.chunk(params, 2, dim=1)
        self.mean = mean
        # Prevent numerical instability
        self.logvar = torch.clamp(logvar, min=-10, max=10)
        self.std = torch.exp(0.5 * logvar)

    def sample(self, generator: Optional[torch.Generator]=None):
        sample = torch.randn(self.mean.shape, device=self.params.device, generator=generator)
        return self.mean + self.std * sample

class MiniAttention(nn.Module):
    def __init__(self, channels, norm_groups):
        super().__init__()
        self.norm = nn.GroupNorm(norm_groups, channels)
        self.act = nn.SiLU()
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        self.project = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor):
        input = x

        # Normalize
        x = self.act(self.norm(x))

        # Reshape and transpose
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).transpose(1, 2)

        # Calculate attention scores
        query: torch.Tensor = self.query(x)
        key: torch.Tensor = self.key(x)
        value: torch.Tensor = self.value(x)

        scores = torch.baddbmm(
            torch.empty(
                query.shape[0],
                query.shape[1],
                query.shape[1],
                dtype=x.dtype,
                device=x.device,
            ),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=1,
        )

        probs = torch.softmax(scores.float(), dim=-1).type(scores.dtype)
        x = torch.bmm(probs, value)
        x = self.project(x)
        x = x.transpose(1, 2).view(b, c, h, w)

        return x + input

class MiniMid(nn.Module):
    def __init__(self, channels: int, norm_groups: int, num_layers: int=1):
        super().__init__()
        self.layers = nn.ModuleList([
            MiniResNet(channels, channels, norm_groups)
        ])
        for i in range(num_layers - 1):
            self.layers.append(MiniAttention(channels, norm_groups))
            self.layers.append(MiniResNet(channels, channels, norm_groups))

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)

        return x

class MiniResNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm_groups: int):
        super().__init__()
        self.act = nn.SiLU()

        self.norm = nn.GroupNorm(norm_groups, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor):
        input = x
        x = self.act(self.norm(x))
        x = self.act(self.conv(x))

        return x + self.shortcut(input)

class MiniDownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm_groups: int, num_layers: int=1, down: bool=True):
        super().__init__()
        self.layers = nn.ModuleList([
            MiniResNet(in_channels, out_channels, norm_groups)
        ])
        for i in range(num_layers - 1):
            self.layers.append(MiniResNet(out_channels, out_channels, norm_groups))

        if down:
            self.down = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        else:
            self.down = nn.Identity()

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)

        return self.down(x)

class MiniUpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm_groups: int, num_layers: int=1, up: bool=True, use_transpose: bool=False):
        super().__init__()
        self.layers = nn.ModuleList([
            MiniResNet(in_channels, out_channels, norm_groups)
        ])
        for i in range(num_layers - 1):
            self.layers.append(MiniResNet(out_channels, out_channels, norm_groups))

        if up:
            if not use_transpose:
                self.up = nn.Upsample(scale_factor=2, mode='nearest')
            else:
                self.up = nn.ConvTranspose2d(out_channels, out_channels, 3, 2, 1, 1)
        else:
            self.up = nn.Identity()

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)

        return self.up(x)

class MiniEncoder(nn.Module):
    def __init__(self, in_channels: int, steps: List[int], norm_groups: int, layers_per_block: int=1):
        super().__init__()
        prev = steps[0]
        self.conv = nn.Conv2d(in_channels, prev, 3, 1, 1)
        self.down = nn.ModuleList()

        for i, step in enumerate(steps[1:]):
            last = i == len(steps) - 2
            self.down.append(MiniDownBlock(prev, step, norm_groups, num_layers=layers_per_block, down=not last))
            prev = step

        self.mid = MiniMid(prev, norm_groups, num_layers=layers_per_block)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)

        for layer in self.down:
            x = layer(x)

        x = self.mid(x)

        return x

class MiniDecoder(nn.Module):
    def __init__(self, in_channels: int, steps: List[int], norm_groups: int, layers_per_block: int=1):
        super().__init__()

        prev = steps[0]
        self.conv = nn.Conv2d(in_channels, prev, 3, 1, 1)
        self.mid = MiniMid(prev, norm_groups, num_layers=layers_per_block)

        self.up = nn.ModuleList()

        for i, step in enumerate(steps[1:]):
            first = i == 0
            self.up.append(MiniUpBlock(prev, step, norm_groups, num_layers=layers_per_block, up=not first))
            prev = step

        self.norm = nn.GroupNorm(norm_groups, prev)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.mid(x)
        for layer in self.up:
            x = layer(x)

        x = self.act(self.norm(x))

        return x

class MiniVae(nn.Module):
    class SampleMode(Enum):
        RANDOM = 0
        MEAN = 1

    def __init__(self, in_channels: int, steps: List[int], norm_groups: int, latent_channels: int, layers_per_block: int=1, generator: Optional[torch.Generator]=None):
        super().__init__()
        if len(steps) == 0:
            raise ValueError('steps must be non-empty')
            
        self.encoder = MiniEncoder(in_channels, steps, norm_groups, layers_per_block=layers_per_block)
        self.norm = nn.GroupNorm(norm_groups, steps[-1])
        self.act = nn.SiLU()
        self.quant_conv = nn.Conv2d(steps[-1], latent_channels * 2, 3, 1, 1)

        self.decoder = MiniDecoder(latent_channels, list(reversed(steps)), norm_groups, layers_per_block=layers_per_block)
        self.conv_out = nn.Conv2d(steps[0], in_channels, 3, 1, 1)
        self.generator = generator
        self.sample_mode = MiniVae.SampleMode.RANDOM

    def set_mean_mode(self):
        self.sample_mode = MiniVae.SampleMode.MEAN

    def set_sample_mode(self):
        self.sample_mode = MiniVae.SampleMode.RANDOM

    def toggle_sample_mode(self):
        if self.sample_mode == MiniVae.SampleMode.RANDOM:
            self.sample_mode = MiniVae.SampleMode.MEAN
        else:
            self.sample_mode = MiniVae.SampleMode.RANDOM

    def encode(self, x: torch.Tensor):
        x = self.encoder(x)
        x = self.act(self.norm(x))
        x = self.quant_conv(x)
        posterior = MiniDist(x)

        return posterior

    def decode(self, x: torch.Tensor):
        x = self.decoder(x)
        x = self.conv_out(x)

        return x

    def forward(self, x: torch.Tensor):
        posterior = self.encode(x)
        if self.sample_mode == MiniVae.SampleMode.RANDOM:
            sample = posterior.sample(self.generator)
        else:
            sample = posterior.mean
        reconstructed = self.decode(sample)

        return reconstructed

vae = MiniVae(1, [8, 16, 32], norm_groups=4, latent_channels=4, layers_per_block=2).to('cuda')
vae.set_mean_mode()

tiles = Image.open('input/all_uniques.png').convert('L')
output = Image.new('L', (tiles.width, tiles.height))
control = Image.new('L', (tiles.width, tiles.height))
# Combine tiles into a single tensor, with each tile as a batch
# Keep the first 300 for validation
tileset = None
validations = None
i = 0
for y in range(0, tiles.height, 16):
    for x in range(0, tiles.width, 16):
        tile = tiles.crop((x, y, x + 16, y + 16))
        tile = transforms.ToTensor()(tile)
        tile = tile.unsqueeze(0)
        if i < 300:
            if validations is None:
                validations = tile
            else:
                validations = torch.cat((validations, tile), 0)
            i += 1
        else:
            if tileset is None:
                tileset = tile
            else:
                tileset = torch.cat((tileset, tile), 0)

tileset = tileset.to('cuda')
validations = validations.to('cuda')
 
size = (tiles.height // 16) * (tiles.width // 16) - 300
loader = DataLoader(tileset, batch_size=size, shuffle=True)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

epochs = 1000
losses = []
tx, ty = 0, 0
cx, cy = 0, 0
outputs = []
for epoch in range(epochs):
    if epoch == epochs - 1:
        vae.set_mean_mode()
    elif epoch > 10:
        vae.toggle_sample_mode()

    for image in loader:
        optimizer.zero_grad()
        result = vae(image)
        loss = loss_fn(result, image)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if epoch == epochs - 1:
            reconstructeds = result.squeeze(0)
            # Slice the image batches back into individual tiles
            for reconstructed, original in zip(reconstructeds, image):
                outputs.append((reconstructed, original))
        
    progress(epoch + 1, epochs, prefix='Epoch:', suffix='Complete', length=50)

validation_loader = DataLoader(validations, batch_size=300, shuffle=True)
validation_losses = []
for image in validation_loader:
    result = vae(image)
    loss = loss_fn(result, image)
    validation_losses.append(loss.item())
    reconstructeds = result.squeeze(0)
    for reconstructed, original in zip(reconstructeds, image):
        outputs.append((reconstructed, original))

plt.style.use('fivethirtyeight')
plt.xlabel('Iterations')
plt.ylabel('Recent Loss')
plt.plot(losses[-100:])
plt.savefig(f'output/loss1.png')
plt.close()

plt.style.use('fivethirtyeight')
plt.xlabel('Iterations')
plt.ylabel('Overall Loss') 
plt.plot(losses)
plt.savefig(f'output/loss2.png')
plt.close()

plt.style.use('fivethirtyeight')
plt.xlabel('Iterations')
plt.ylabel('Validation Loss')
plt.plot(validation_losses)
plt.savefig(f'output/loss3.png')
plt.close()
print(validation_losses)

output = Image.new('L', (tiles.width, tiles.height * 2))
x, y = 0, 0
for reconstructed, original in outputs:
    original = transforms.ToPILImage()(original)
    #reconstructed = quanitize(transforms.ToPILImage()(reconstructed))
    reconstructed = transforms.ToPILImage()(reconstructed)
    output.paste(original, (x, y))
    output.paste(reconstructed, (x, y + 16))
    x += 16
    if x >= tiles.width:
        x = 0
        y += 32

output.save(f'output/output.png')


# tile = test.crop((0, 0, 16, 16))
# tile = transforms.ToTensor()(tile)
# tile = tile.unsqueeze(0)
# result = vae(tile)
# print(f'encoding.shape = {result.shape}')
# output = result.squeeze(0)
# output = transforms.ToPILImage()(output)
# output.save('output/test.png')