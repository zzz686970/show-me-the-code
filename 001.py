#!-*-coding:utf-8-*-
## addd numbers into a pic
from PIL import Image, ImageDraw, ImageFont


ascii_char = list("$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. ")
length = len(ascii_char)

def add_num(num, fill, font_name):
    im = Image.open("/Users/macbook/Documents/跑步/sun.jpeg")
    xsize, ysize = im.size
    draw = ImageDraw.Draw(im)
    text = str(num)
    font = ImageFont.truetype(font_name, xsize // 6)
    #im = im.resize((int(xsize*0.5), int(ysize) * 0.5))
    ## left, upper, right, lower
    draw.ellipse(((xsize * 0.85, 0), (xsize, xsize * 0.15)), fill="red", outline="red")
    # draw.ellipse((xsize//2, xsize//2, ysize, ysize), fill = 'red', outline= 'red')
    draw.text((xsize * 0.88 , -10), text, fill, font)
    txt = convert(im, length)
    f = open('/Users/macbook/Documents/跑步/convert.txt', 'w')
    f.write(txt)
    f.close()
    im.save("/Users/macbook/Documents/跑步/sun_out.jpeg")

def convert(im, length):
    im = im.convert('L')
    txt = ""
    for i in range(im.size[1]):
        for j in range(im.size[0]):
            gray = im.getpixel((j,i))
            unit = 256.0/ length
            txt += ascii_char[int(gray/unit)]
        txt += '\n'
    return txt

def convert1(im, length):
    txt = ""
    for i in range(im.size[1]):
        for j in range(im.size[0]):
            r,g,b = im.getpixel((j,i))
            gray = int(r * 0.299 + g * 0.587 + b * 0.114)  # 通过灰度转换公式获取灰度
            unit = (256.0 + 1)/ length
            ## 获取坐标
            txt += ascii_char[int(gray/unit)]
        txt += '\n'
    return txt

num = 1
fill= 'blue'
font_name = '/Library/Fonts/Arial.ttf'
add_num(num, fill, font_name)


# from PIL import Image, ImageDraw, ImageFont,ImageFilter
# import random
#
# def randChar():
#     return chr(random.randint(65, 90)  or random.randint(97, 122))
#
# def randText():
#     text = []
#     text.append(random.randint(97,122))
#     text.append(random.randint(65,90))
#     text.append(random.randint(48,57))
#     return chr(text[random.randint(0,2)])
#
# def randColor2():
#     return (random.randint(32, 127), random.randint(32, 127),random.randint(32, 127))
#
# def randColor():
#     return (random.randint(64, 255), random.randint(64, 255), random.randint(64, 255))
#
# width = 60 * 4
# height = 60
# image = Image.new('RGB', (width, height), (255, 255, 255))
#
# font = ImageFont.truetype('/Library/Fonts/Arial.ttf', 36)
# draw = ImageDraw.Draw(image)
#
# for x in range(width):
#     for y in range(height):
#         draw.point((x, y), fill = randColor())
#
# for t in range(4):
#     draw.text((60 * t + 10, 10), randText(),  font=font, fill=randColor2())
#
# image = image.filter(ImageFilter.BLUR)
# image.save('/Users/macbook/Downloads/code.jpg', "jpeg")
# #
# import random
# import string
#
# letters = [i for i in string.ascii_lowercase]
# random_list = random.sample(letters, 15)
#
# print (random_list)