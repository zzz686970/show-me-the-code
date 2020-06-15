import urllib.request
import sys
import zipfile
from io import StringIO

# zobj = StringIO()
# zobj.write(str(urllib.request.urlopen("http://pythonchallenge.com/pc/def/channel.zip").read()))
# z = zipfile.ZipFile("channel.zip", 'r')

# filenum = "90052"
# lcomment = []

# while True:
# 	if sys.version_info >= (3, 0):
# 	    if filenum.isdigit():
# 	        filename = filenum + '.txt'
# 	        for f_name in z.namelist():
# 	        	print(z.getinfo(f_name).comment)
# 	        # print(z.getinfo(filename).comment)
# 	        # lcomment.append(z.getinfo(filename).comment)
# 	        	info = z.read(f_name)
# 	        	print(info.split(' ')[-1])
# 	        	filenum = info.split(' ')[-1]
# 	    else:
# 	        break
# z.close()
# print (''.join(str(lcomment)))





# import zipfile

# def before(value, a):
#     # Find first part and return slice before it.
#     pos_a = value.find(a)
#     if pos_a == -1: return ""
#     return value[0:pos_a]

# with zipfile.ZipFile('channel.zip', 'r') as z:
#     zips = z.infolist()
#     files = {}
#     for zip in zips:
#         content = z.read(zip.filename).decode('utf-8')
#         files[before(zip.filename, '.txt')] = (content[content.rfind(
#             ' ')+1:], bytes(zip.comment).decode('utf-8'))
#     print(files)
#     pointer = files['90052']
#     while True:
#         print(files[pointer[0]][1], end='')
#         pointer = files[pointer[0]]


import zipfile
import re
import sys


def get_file():
    if sys.version[0] == '3':
        import urllib.request
        urllib.request.urlretrieve('http://www.pythonchallenge.com/pc/def/channel.zip','./channel.zip')
    else:
        import urllib
        urllib.urlretrieve('http://www.pythonchallenge.com/pc/def/channel.zip','./channel.zip')

  
get_file()
zip_file=zipfile.ZipFile('./channel.zip')

file_list=[]
next_str='90052'
file_content=zip_file.read('%s.txt' % next_str).decode('utf-8')
while next_str:
    file_list.append("%s.txt" % next_str)
    next_str=re.compile('Next nothing is ([0-9]+)').match(file_content)
    if next_str:
        next_str=next_str.groups()[0]
        file_content=zip_file.read('%s.txt' % next_str).decode('utf-8')
        # print( file_content)
    else:
        break
    
# print: Collect the comments.
print ("".join([zip_file.getinfo(i).comment.decode('utf-8') for i in file_list]))
