import tensorflow as tf
import numpy as np
from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.datasets import mnist
from keras import regularizers
import matplotlib.pyplot as plt

inputImg = Input(shape=(28,28,1))

x = Convolution2D(16,(3,3), activation='relu', border_mode='same')(inputImg)
x = MaxPooling2D((2,2),border_mode='same')(x)
x = Convolution2D(8,(3,3), activation='relu', border_mode='same')(x)
x = MaxPooling2D((2,2),border_mode='same')(x)
x = Convolution2D(8,(3,3), activation='relu', border_mode='same')(x)
encoded = MaxPooling2D((2,2),border_mode='same')(x)

x = Convolution2D(8,(3,3),activation='relu',border_mode='same')(encoded)
x = UpSampling2D((2,2))(x)
x = Convolution2D(8,(3,3),activation='relu',border_mode='same')(x)
x = UpSampling2D((2,2))(x)
x = Convolution2D(16,(3,3),activation='relu')(x)
x = UpSampling2D((2,2))(x)
decoded = Convolution2D(1,(3,3),activation='sigmoid',border_mode='same')(x)

autoencoder = Model(inputImg,decoded)
autoencoder.compile(optimizer = 'adam', loss = 'binary_crossentropy')

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.
x_train = np.reshape(x_train, (len(x_train),28,28,1))
x_test = np.reshape(x_test, (len(x_test),28,28,1))

autoencoder.fit(x_train,x_train,
        nb_epoch=50,
        batch_size=128,
        shuffle=True,
        validation_data=(x_test,x_test)
)

decoded_imgs = autoencoder.predict(x_test)

n = 10
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2,n,i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2,n,i+1 + n)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

encoder = Model(input_img,encoded)
encoded_imgs = encoder.predict(x_test[:n])
plt.figure(figsize(20,8))
for i in range(n):
    for j in range(8):
        ax = plt.subplot(8,n,j*n+i+1)
        plt.imshow(encoded_imgs[i][j],interpolation='none')
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
plt.show()
