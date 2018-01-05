import numpy as np

# define entropy
'''
    labels: array with shape (n,), each entry represents a label for 1 sample
    nCls  : number of classes (if your labels are 0 and 1, then nCls is 2)
'''
def entropy(labels,nCls):
    res = 0
    total = labels.shape[0]
    for cls in range(0,nCls):
        p = labels[labels==cls].shape[0]
        pi = float(p)/float(total)
        if pi == 0:
            continue
        res -= pi*np.log2(pi)
    return res


def cov(vect1,vect2):
    if vect1.shape != vect2.shape:
        raise Exception('shapes of two vectors should be the same! ')
    return np.dot(vect1,np.transpose(vect2))/vect1.shape[0]

# pearson correlation coefficient
def pcc(vect1,vect2):
    # calc covariance of two vectors
    covariance = cov(vect1-np.mean(vect1),vect2-np.mean(vect2))
    var1 = np.var(vect1)
    var2 = np.var(vect2)
    return covariance/np.sqrt(var1*var2)


# initialize an array
labels = np.array([0,0,1,1,1])

feature1 = np.array([2,1,3])
feature2 = np.array([1,3,2])

print 'pearson correlation coefficient: '+str(pcc(feature1,feature2))

# print its shape
print 'the shape of this array is '+str(labels.shape)

# print count of 1
cnt_1 = labels[labels==1].shape[0]
print 'there are '+str(cnt_1)+' ones in the array'

# print count of 0
cnt_0 = labels[labels==0].shape[0]
print 'there are '+str(cnt_0)+' zeros in the array'

# print mean
meanArr = np.mean(labels)
print 'mean of the array is '+str(meanArr)

# print median
medianArr = np.median(labels)
print 'median of the array is '+str(medianArr)

# print stdvar
stdvar = np.std(labels)
print 'stdvar of the array is '+str(stdvar)

# print entropy
print 'entropy of this array is '+str(entropy(labels,2))



