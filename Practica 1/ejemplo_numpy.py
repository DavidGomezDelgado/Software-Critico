# https://www.geeksforgeeks.org/python-numpy/


import numpy as np
import matplotlib.pyplot as plt
 
# Creating a rank 1 Array
arr = np.array([1, 2, 3])
print("Array with Rank 1: \n",arr)
 
# Creating a rank 2 Array
arr = np.array([[1, 2, 3],
                [4, 5, 6]])
#arr.shape me dice la forma del array
print("Prueba shape: ", arr.shape)
print("Array with Rank 2: \n", arr)

plt.scatter(x=arr[0],y=arr[1])
plt.show()

input('indexing')
# Initial Array
arr = np.array([[-1, 2, 0, 4],
                [4, -0.5, 6, 0],
                [2.6, 0, 7, 8],
                [3, -7, 4, 2.0]])
print("Initial Array: ")
print(arr)
 
# Printing a range of Array
# with the use of slicing method
sliced_arr = arr[:2, 1:3] #me quedo con las dos primeras filas y las columnas de la 1 a la 2 (no cuenta la 3)

#[[ 2.   0. ]
 #[-0.5  6. ]]
print ("Array with first 2 rows and"
    " columns 1 to 2:\n", sliced_arr)

input('basic operations on single array')
 
# Defining Array 1
a = np.array([[1, 2],
              [3, 4]])
 
# Defining Array 2
b = np.array([[4, 3],
              [2, 1]])
               
# Adding 1 to every element
print ("Adding 1 to every element:", a + 1)
 
# Subtracting 2 from each element
print ("\nSubtracting 2 from each element:", b - 2)
 
# sum of array elements
# Performing Unary operations
print ("\nSum of all array "
       "elements: ", a.sum())
 
# Adding two arrays
# Performing Binary operations
c = a + b
print ("\nArray sum:\n", c)

# Reshape example
input('reshape and error')
c=c.reshape(-1)

# Calculating error
# d is generated from c (first parameter) and a small variation
d = np.random.normal(c, c * 0.1)

print ("Array c:", c)
print("Array d:", d)

# Average error
error = np.mean(np.abs(c - d))

print("Medium error:", error)

plt.plot(c, label="Array c")
plt.plot(d, label="Array d")
plt.xlabel("Index")
plt.ylabel("Value")
plt.title("Array comparison")
plt.legend()
plt.show()
 
