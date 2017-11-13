import random

def randText():
    text = []
    text.append(random.randint(97,122))
    text.append(random.randint(65,90))
    text.append(random.randint(48,57))
    return chr(text[random.randint(0,2)])

if __name__ == '__main__':
    name = ''
    list1 = []
    for i in range(200):
        for j in range(16):
            name += randText()
        list1.append(name)
        name = ''

    with open('random_coupon.txt', 'w') as f:
        for i in range(len(list1)):
            f.write(list1[i] + '\n')