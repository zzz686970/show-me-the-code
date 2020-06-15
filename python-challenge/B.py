"""
Problem : Min Product Array

The task is to find the minimum sum of Products of two arrays of the same size, given that k modifications are allowed on the first array. In each modification, one array element of the first array can either be increased or decreased by 2. 

Note- the product sum is Summation (A[i]*B[i]) for all i from 1 to n where n is the size of both arrays 

Input Format: 
First line of the input contains n and k delimited by whitespace
Second line contains the Array A (modifiable array) with its values delimited by spaces
Third line contains the Array B (non-modifiable array) with its values delimited by spaces

Output Format: 

Output the minimum sum of products of the two arrays 

Constraints:

1 ≤ N ≤ 10^5
0 ≤ |A[i]|, |B[i]| ≤ 10^5
0 ≤ K ≤ 10^9


Sample Input and Output

SNo.	Input	Output
1	
3 5
1 2 -3
-2 3 -5

-31
2	
5 3
2 3 4 5 4
3 4 2 3 2

25


Explanation for sample 1: 

Here total numbers are 3 and total modifications allowed are 5. So we modified A[2], which is -3 and increased it by 10 (as 5 modifications are allowed). Now final sum will be 
(1 * -2) + (2 * 3) + (7 * -5) 
-2 + 6 - 35 
-31 

-31 is our final answer. 

Explanation for sample 2: 

Here total numbers are 5 and total modifications allowed are 3. So we modified A[1], which is 3 and decreased it by 6 (as 3 modifications are allowed). 
Now final sum will be 
(2 * 3) + (-3 * 4) + (4 * 2) + (5 * 3) + (4 * 2) 
6 - 12 + 8 + 15 + 8 
25 

25 is our final answer. 
"""
# a1 = [1,2,-3]
# b1 = [-2,3,-5]
# total_version = []
# from functools import reduce
# def min_product_array(N=5, k = 4):
# 	min_result = 0
# 	for j in range(k):
# 		for i in range(N):
# 			modified = a1[i] +2
# 		for ele in list(zip(a1, b1)):
# 			result = sum(reduce(lambda x,y: x*y, ele ))
# 			if result < min_result:
# 				min_result = result
a1 = [1,2,-3]
b1 = [-2,3,-5]
max_diff, min_result = 0, 0
for i in range(n):
	for ele in list(zip(a1, b1)):
		temp = 
		result = sum(reduce(lambda x,y: x*y, ele ))
		if (result < 0 and b1[i] < 0):
			temp = (a1[i] + 2 * k) * a1[i]
		elif (result < 0 and a1[i] < 0):
			temp = (a1[i] - 2 * k) * b1[i]
		elif(result > 0 and a1[i] < 0):
			temp = (a1[i] + 2*k) *b1[i]
		elif (result > 0 and a1[i] > 0):
			temp = (a1[i]  - 2*k) * b1[i]





