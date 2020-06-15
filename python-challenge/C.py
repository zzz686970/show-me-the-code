"""
Problem : Consecutive Prime Sum

Some prime numbers can be expressed as Sum of other consecutive prime numbers. 
For example 

5 = 2 + 3 
17 = 2 + 3 + 5 + 7 
41 = 2 + 3 + 5 + 7 + 11 + 13 

Your task is to find out how many prime numbers which satisfy this property are present in the range 3 to N subject to a constraint that summation should always start with number 2. 

Write code to find out number of prime numbers that satisfy the above mentioned property in a given range. 

Input Format: 

First line contains a number N 

Output Format: 

Print the total number of all such prime numbers which are less than or equal to N. 

Constraints:
1. 2<N<=12,000,000,000

Sample Input and Output

SNo.	Input	Output	Comment
1	20	2	
(Below 20, there are 2 such numbers: 5 and 17).
5=2+3
17=2+3+5+7
2	15	1	

"""


N = raw_input("Enter a number:")



# prime_list = []
# for ele in range(20):
# 	if ele > 1:
# 		for factor in range(2, ele):
# 			if ele % factor == 0:
# 				break
# 		else:
# 			prime_list.append(ele)

def return_primes(upto=100):
    primes = []
    sieve = set()
    for i in range(2, upto+1):
        if i not in sieve:
            primes.append(i)
            sieve.update(range(i, upto+1, i))
    return primes

def find_prime_sum_consecutive_prime(x):
	"""
	given x, an integer, return the prime that is the sum of consecutive prime numbers under x
	"""

	prime = return_primes(x)
	length = len(prime)
	prime_set = set(prime)
	max_prime = prime[-1]
	result = []
	min_length = 1
	for start in range(0, length):
		for window_len in range(start+min_length, length -start +1 ):
			check_prime = prime[start:window_len]
			s = sum(check_prime)
			if s in prime_set and len(check_prime) > 1:
				min_length = len(check_prime)
				result.append(s)
			elif s > max_prime:
				break
	return len(result)

if __name__ == '__main__':
	x = find_prime_sum_consecutive_prime(N)
	print(x)
