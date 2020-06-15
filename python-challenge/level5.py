import urllib, re, sys
import pickle

url = "http://www.pythonchallenge.com/pc/def/banner.p"
list1 = []
try:
	if sys.version_info >= (3,0):
		import urllib.request
		resp = urllib.request.Request(url)
		data = pickle.load(urllib.request.urlopen(resp))
		list1 = data.copy()
	else:
		data = urllib.urlopen(url)
except:
	print("Error")

# print(list1)
for ele in list1:
	print("".join([e[1] *e[0] for e in ele]))