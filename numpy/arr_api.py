import numpy as np

# initialize an array
shape = (1,2) 
arr_0 = np.zeros(shape)
print arr_0

arr_1 = np.ones(shape)
print arr_1

square = np.ones((2,2))
square[0,0] = 0

# operations

# 1. attributes
# shape of array (a tuple of integers)
print arr_0.shape


# 2. transpose
arr_0T = np.transpose(arr_0)
print arr_0T

arr_1T = arr_0.T
print arr_1T

# 3. indexing
# get some columns or rows
print arr_0[:,0]
print arr_0[:,0:2]

# 4. mathematical operations
# add
print arr_0+arr_1

# minus
print arr_0-arr_1

# multiplication (*)
print arr_0*arr_1

# dot production
print arr_0.dot(arr_1.T)

# inverse
inverse = np.linalg.inv(square)
print inverse

# divide (dot product with its inverse)
# result should be E if multiply its inverse
divided = square.dot(inverse)
print divided

# 5. broadcasting

# add array with shape(1,2) and an integer = add [integer,integer] to the array 
print arr_0+1

# add array with shape(1,2) and another array with shape(2,2) = repeat array(1,2) to (2,2) and then added to array(2,2)
print arr_0+square

# 6. stat
# mean
print arr_0.mean()

# variance
print arr_0.var()

# std var
print square.std()

# covariance
def cov(vect1,vect2):
    if vect1.shape != vect2.shape:
        raise Exception('shapes of two vectors should be the same! ')
    return np.dot(vect1,np.transpose(vect2))[0][0]/vect1.shape[0]


print cov(arr_0,arr_1)



#sigmoid
def sigmoid(input):
    return 1/(1+np.power(np.e,input))

print sigmoid(arr_1)


