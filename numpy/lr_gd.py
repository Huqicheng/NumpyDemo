import numpy as np
import matplotlib.pyplot as plt

def readFile(path):
    a = np.loadtxt(path)
    return a

def splitData(data):
    nDim = data.shape[1]-1
    data_x = data[:,0:nDim]
    data_y = data[:,nDim]
    return data_x,data_y

def sigmoid(input):
    return 1/(1+np.power(np.e,input))


def calcGradient(w,dataX,dataY):
    wT = np.transpose(w)
    # Y - sigmoid(w.T*X)
    loss = dataY - sigmoid(np.dot(wT,dataX))
    # loss*X.T
    grad = np.dot(loss,np.transpose(dataX))
    return grad


def calcGradientWithL2(w,dataX,dataY):
    wT = np.transpose(w)
    # Y - sigmoid(w.T*X)
    loss = dataY - sigmoid(np.dot(wT,dataX))
    # loss*X.T
    grad = np.dot(loss,np.transpose(dataX))+wT
    return grad


'''
    batch gradient descent
'''
def BatchGD(dataX,dataY,learning_rate=0.05,calcGrad=calcGradient):
    nDim = dataX.shape[0]
    w = np.zeros((nDim,1))
    preW = w
    while (True):
        # calc gradient
        grad = calcGrad(w,dataX,dataY)
        
        # w = w - lr * grad
        w = w - np.transpose(learning_rate*grad)
        
        # convergence
        if(np.linalg.norm(w-preW)<=0.001):
            break
        
        preW = w
    return w


if __name__=="__main__":
    #  output of readFile is an ndarray
    data = readFile('data2.txt')
    
    # split data
    data_x,data_y = splitData(data)
    data_x = np.transpose(data_x)
    
    w = BatchGD(data_x,data_y,calcGrad=calcGradientWithL2)
    res = (sigmoid(np.dot(w.T,data_x)))[0]
    print "training with L2 Regularization \n" + str(w)
    print res
    
    w2 = BatchGD(data_x,data_y,calcGrad=calcGradient)
    res2 = (sigmoid(np.dot(w2.T,data_x)))[0]
    print "training without L2 Regularization \n" + str(w2)
    print res2







    



