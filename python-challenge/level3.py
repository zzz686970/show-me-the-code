import string
import re
import urllib.request
## 因为注释写了中文，要么在开头声明，要么用py3
#
# 得到的匹配字符串，但只需要取出其中的小写字母，因此用括号

# get string data
data = urllib.request.urlopen("http://www.pythonchallenge.com/pc/def/equality.html").read().decode()
pattern = re.compile("[^A-Z]+[A-Z]{3}([a-z])[A-Z]{3}[^A-Z]+")

print("".join(re.findall(pattern, data)))
# linkedlist


