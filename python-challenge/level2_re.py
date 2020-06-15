import re
import urllib.request

html = urllib.request.urlopen("http://www.pythonchallenge.com/pc/def/ocr.html").read().decode()
## 匹配换行符
data = re.findall("<!--(.*?)-->", html, re.DOTALL)[-1]
print("".join(re.findall("[a-zA-Z]",data)))