from PIL import Image

input = Image.open('res/gridded/zelda_oracle_of_seasons_dirty.png')
w = input.width - (input.width // 160) - 1
h = input.height - (input.height // 128)
output = Image.new('RGB', (w, h))
oy = 0
for iy in range(1, input.height, 128 + 1):
    ox = 0
    for ix in range(1, input.width, 160 + 1):
        panel = input.crop((ix, iy, ix + 160, iy + 128))
        output.paste(panel, (ox, oy, ox + 160, oy + 128))
        ox += 160
    oy += 128

output.save('res/colorized/zelda_oracle_of_seasons.png')