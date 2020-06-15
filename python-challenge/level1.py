## dict init: info = {}  or info = dict()

## 
import string
import sys
# print(sys.version)

# chr(ord('a')+2)  --> 'c'

from string import *
a=ascii_lowercase
"map".translate(maketrans(a,a[2:]+a[:2]))


encrypted_data = "abcdefghijklmnopqrstuvwxyz"

decoded_data = "cdefghijklmnopqrstuvwxyzab"

decode_dict = {}
for i in zip(encrypted_data, decoded_data):
	decode_dict[i[0]] = i[1]
# print(decode_dict['z']) 


puzzle = "g fmnc wms bgblr rpylqjyrc gr zw fylb. rfyrq ufyr amknsrcpq ypc dmp. bmgle gr gl zw fylb gq glcddgagclr ylb rfyr'q ufw rfgq rcvr gq qm jmle. sqgle qrpgle.kyicrpylq() gq pcamkkclbcb. lmu ynnjw ml rfc spj."

key = 'map'

def map_it(s, r):
	for char in s:
		o = ord(char)
		if o in range(97, 123):
			char = chr((o-97 + r) % 26 + 97)

		yield char

print("".join(c for c in map_it(key, 2)))

def translate(puzzle):
	result = ""
	for ele in puzzle:
		if ele in string.ascii_letters:
			if ele.lower() == 'y':
				result += 'a'
			elif ele.lower() == 'z':
				result += 'b'
			else:
				result += chr(ord(ele) + 2)
		else:
			result += ele
	return result

print(translate(key))
# for ele in puzzle:

# 	print(ele)