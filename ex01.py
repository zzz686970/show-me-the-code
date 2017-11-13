'''
    add numbers into a picture
'''
ascii_char = list("$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. ")
length = len(ascii_char)

from PIL import Image, ImageDraw, ImageFont

# def add_number(text, fill, font_name):
#     im = Image.open('/Users/macbook/Downloads/ascii_dora.png')
#     xsize, ysize = im.size
#     im = im.resize((xsize // 2, ysize //2 ), Image.NEAREST)
#     text = str(text)
#     draw = ImageDraw.Draw(im)
#     font = ImageFont.truetype(font_name, xsize // 6)
#     draw.ellipse(((xsize * 0.85, 0), (xsize, xsize * 0.15)), fill='blue', outline='red')
#     draw.text((xsize * 0.88, -10), text, fill, font)
#     txt = convert(im, length)
#     with open('/Users/macbook/Downloads/ascii_dora.txt', 'w') as f:
#         f.write(txt)
#     im.save('/Users/macbook/Downloads/ascii_dora.png_1.png')

def convert(im, length):
    # im = im.convert('L')
    txt = ""
    for i in range(im.size[1]):
        for j in range(im.size[0]):
            r, g, b = im.getpixel((j, i))[:3]
            # gray = im.getpixel((j, i))
            gray = int((19595 * r + 38469 * g + 7472 * b) >> 16)
            unit = 256.0 / length
            txt += ascii_char[int(gray/unit)]
        txt += '\n'
    return txt

if __name__ == '__main__':
    font_name = '/Library/Fonts/Arial.ttf'
    im = Image.open('/Users/macbook/Downloads/ascii_dora2.png')
    xsize, ysize = im.size
    im = im.resize((xsize // 2, ysize // 2 ), Image.NEAREST)
    draw = ImageDraw.Draw(im)
    txt = convert(im, length)
    with open('/Users/macbook/Downloads/ascii_dora2.txt', 'w') as f:
        f.write(txt)
    im.save('/Users/macbook/Downloads/ascii_dora2.png')
