import random

def randText():
    text = []
    text.append(random.randint(97,122))
    text.append(random.randint(65,90))
    text.append(random.randint(48,57))
    return chr(text[random.randint(0,2)])

def randText2():
    f = random.choice('qwertyuiop')
    pwd = ''.join([i for i in f])
    return pwd

if __name__ == '__main__':
    print(randText2())