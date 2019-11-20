import numpy as np
from keras.datasets import mnist
from PIL import Image
import ActivationFunction as acfunc
def InitNetwork():
    
def predict(network,x):
    W1,W2,W3 = network['W1'],network['W2'],network['W3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']

    a1 = np.dot(x,W1) + b1
    z1 = acfunc.Sigmoid(a1)
    a2 = np.dot(z1,W2) + b2
    z2 = acfunc.Sigmoid(a2)
    a3 = np.dot(z2,W3) + b3
    y = acfunc.SoftMax(a3)

    return y

(x_image, x_label),(test_image,test_label) = mnist.load_data()
network = InitNetwork()

accuracy_cnt=0
for i in range(len(x)):
    y = predict(network,x[i])
    p = np.argmax(y)
    if p == t[i]:
        accuracy_cnt += 1
    
print("Accuracy:" + str(float(accuracy_cnt)/len(x)))

batch_size = 100
accuracy_cnt
for i in range(0,len(x),batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network,x_batch)
    p = np.argmax(y_batch,axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuract:" +str(float(accuracy_cnt)/len(x)))