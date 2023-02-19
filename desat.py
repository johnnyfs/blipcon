from PIL import Image

input = Image.open('res/colorized/zelda_oracle_of_seasons_fixed2.png')
input_luminance = input.copy().convert('L')
output = Image.new('L', (input.width, input.height))

# For each tile, arrange the colors in order of luminance
# Fail if there are more than 4 colors
# Convert each color to 0, 84, 170, or 255
for ty in range(0, input.height, 16):
    print(f'row {ty} / {input.height}')
    for tx in range(0, input.width, 16):
        # First pass, collect the luminances
        luminances = set()
        for py in range(16):
            for px in range(16):
                c = input_luminance.getpixel((tx + px, ty + py))
                luminances.add(c)
                if len(luminances) > 4:
                    raise RuntimeError('too many colors at ' + str((tx, ty)))
        # Second pass, convert the colors
        luminances = list(luminances)
        luminances.sort()
        for py in range(16):
            for px in range(16):
                c = input_luminance.getpixel((tx + px, ty + py))
                if c == luminances[0]:
                    output.putpixel((tx + px, ty + py), 0)
                elif c == luminances[1]:
                    output.putpixel((tx + px, ty + py), 84)
                elif c == luminances[2]:
                    output.putpixel((tx + px, ty + py), 170)
                elif c == luminances[3]:
                    output.putpixel((tx + px, ty + py), 255)
                else:
                    raise RuntimeError('wat')
    
output.save('res/desaturated/zelda_oracle_of_seasons.png')
