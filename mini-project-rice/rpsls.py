# The key idea of this program is to equate the strings
# "rock", "paper", "scissors", "lizard", "Spock" to numbers
# as follows:
#
# 0 - rock
# 1 - Spock
# 2 - paper
# 3 - lizard
# 4 - scissors

'''
    exercise url : 
    https://github.com/Lieke22/Interactive-Programming-in-Python-with-Coursera/blob/master/mini%20project%201%20%22Rock-paper-scissors-Lizard-Spock%22.py
'''

import random

def name_to_number(name):
    num_list = range(0,5)
    name_list = ['scissor','paper','rock','lizard','spcok']
    for key, value in zip(name_list, num_list):
        if key == name:
            return value

def number_to_name(num):
    num_list = range(0, 5)
    name_list = ['scissor', 'paper', 'rock', 'lizard', 'spcok']
    for number, result in zip(num_list, name_list):
        if number == num:
            return result

def rpsls(name):
    comp_number = random.randrange(0,5)
    input = name_to_number(name)
    print('I choose {}, computer choose {}'.format(name, number_to_name(comp_number)))
    x = (comp_number - input) % 5
    if x == 1 or x == 2:
        print('computer wins')
    elif x == 3 or x == 4:
        print('player wins')
    else:
        print('a match')

if __name__ == '__main__':
    name_list = ['scissor', 'paper', 'rock', 'lizard', 'spcok']
    player = random.choice(name_list)
    rpsls(player)