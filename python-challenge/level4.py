import urllib, re, time
import sys
import operator


"""
several things when use py3
1. import urllib.request
2. with open urllib.request.Request(url) as f
3. urllib.request.urlopen(req).read().decode("utf-8")

several things for operator
1. eval(x)
2. exec("print " + x)

"""

url = "http://www.pythonchallenge.com/pc/def/linkedlist.php?nothing=%s"

rep = "and the next nothing is (\d+)"

other = re.compile("[0-9]")

ops = { "+": operator.add, "-": operator.sub } # etc.
operation= {
	"multiply": operator.mul,
	"divide": operator.truediv,
	"add": operator.add,
	"minus":operator.sub 
}
num_word = {
	"one": 1,
	"two":2,
	"three":3,
	"four":4,
	"five":5,
	"six":6,
	"seven":7,
	"eight":8,
	"nine": 9,
	"ten": 10
}
# nothing = "16044"
# nothing = "3875" # for test
nothing = "852"
## final 66831
list1 = []
iter_time = 0
a , b = "", ""
while iter_time <= 500:
	try:
		if(sys.version_info >= (3,0)):
			import urllib.request
			req = urllib.request.Request(url % nothing)
			response = urllib.request.urlopen(req)
			source = response.read().decode("utf-8")
		else:			
			source = urllib.urlopen(url % nothing).read()
		print(source)
		if re.search(rep, source) is not None and  isinstance(re.search(rep, source).group(1), str):
			nothing = re.search(rep, source).group(1)
			list1.append(nothing)
		elif re.findall(other, source) == []:
			str2list = source.split(" ")
			for ele in str2list:
				# print(ele)
				if ele.lower() in operation.keys():
					a = ele.lower()
				if ele.lower() in num_word.keys():
					b = num_word[ele.lower()]
			if a is not None and b is not None:
				# print(int(operation[a.lower()](int(list1[-1]), int(b))))
				source = "and the next nothing is " + str(int(operation[a.lower()](int(list1[-1]), int(b))))
				nothing = str(int(operation[a.lower()](int(list1[-1]), int(b))))
				list1.append(nothing)
		iter_time += 1
			
	except:
		print("error happend at page: {}".format(source))
		break

print(len(list1))