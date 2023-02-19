from PIL import Image
import math

paths = [
    'input/dreamboy_village.bmp',
    'input/zgb_uniques.png',
    'input/bm_uniques.png',
    'input/zoos_uniques.png',
]
banks = [ Image.open(path).convert('L') for path in paths ]
#bank = Image.open('res/desaturated/bm.png').convert('L')

# for y in range(zgb.height):
#     for x in range(zgb.width):
#         c = zgb.getpixel((x, y))
#         if c == 96:
#             zgb.putpixel((x, y), 84)
#         elif c == 168:
#             zgb.putpixel((x, y), 170)
#         elif c == 248:
#             zgb.putpixel((x, y), 255)
#         elif c != 0:
#             raise('wat')
# zgb.save('res/zgb2.png')

def image_equal(a, b):
    if a.width != b.width or a.height != b.height:
        return False
    for y in range(a.height):
        for x in range(a.width):
            if a.getpixel((x, y)) != b.getpixel((x, y)):
                return False
    return True

# Extract the unique 16x16 tiles from the bank
tiles = []
padding = 0
for bank in banks:
    for y in range(0, bank.height, 16 + padding):
        print(f'row {y} / {bank.height}')
        for x in range(0, bank.width, 16 + padding):
            tile = bank.crop((x, y, x + 16, y + 16))
            if not any(image_equal(tile, t) for t in tiles):
                tiles.append(tile)
                print(f'adding tile {len(tiles)}')

print(len(tiles))

#bank_out = Image.new('L', (16 * len(tiles), 16))
# width = math.ceil(math.sqrt(len(tiles)))
bank_out = Image.new('L', (16 * 51, 16 * 34))
x, y = 0, 0
for i, tile in enumerate(tiles):
    bank_out.paste(tile, (x, y, x + 16, y + 16))
    x += 16
    if x >= bank_out.width:
        x = 0
        y += 16
bank_out.save('input/all_uniques.png')