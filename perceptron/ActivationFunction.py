import numpy as np

#-----Sigmoid function-----
def Sigmoid(x):
    return 1/(1+np.exp(-x))

def SigmoidGrad(x):
    return Sigmoid(x)*(1.0-Sigmoid(x))
#--------------------------

#-----Step function-----
def Step(x):
    return np.array(x > 0, dtype=np.int)

def StepGrad(x):
    return np.array(x == 0, dtype = np.int)
#-----------------------

#-----ReLU function-----
def ReLU(x):
    return np.maximum(0,x)

def ReLUGrad(x):
    return np.array(x>0,dtype=np.int)
#-----------------------

#-----softmax function-----
def SoftMax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a/sum_exp_a
    return y
#--------------------------