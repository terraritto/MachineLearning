from keras.datasets import mnist
from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from keras.layers.merge import concatenate

# now, we use shallow network
def create_u_net(input):
    inputs = Input((input[0],input[1],1))

    conv1 = Convolution2D(32,(3,3),activation='relu', padding='same')(inputs)
    conv1 = Convolution2D(32,(3,3),activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2,2))(conv1)

    conv2 = Convolution2D(64,(3,3),activation='relu', padding='same')(pool1)
    conv2 = Convolution2D(64,(3,3),activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2,2))(conv2)

    conv3 = Convolution2D(128,(3,3),activation='relu', padding='same')(pool2)
    conv3 = Convolution2D(128,(3,3),activation='relu', padding='same')(conv3)
    """
    pool3 = MaxPooling2D((2,2))(conv3)
    """

    up4 = concatenate([UpSampling2D((2,2))(conv3),conv2])
    conv4 = Convolution2D(64,(3,3),activation='relu', padding='same')(up4)
    conv4 = Convolution2D(64,(3,3),activation='relu', padding='same')(conv4)

    up5 = concatenate([UpSampling2D((2,2))(conv4),conv1])
    conv5 = Convolution2D(32,(3,3),activation='relu', padding='same')(up5)
    conv5 = Convolution2D(32,(3,3),activation='relu', padding='same')(conv5)

    conv6 = Convolution2D(1,(3,3),activation='sigmoid', padding='same')(conv5)

    """
    conv4 = Convolution2D(256,(3,3),activation='relu', padding='same')(pool3)
    conv4 = Convolution2D(256,(3,3),activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D((2,2))(conv4)

    conv5 = Convolution2D(512,(3,3),activation='relu', padding='same')(pool4)
    conv5 = Convolution2D(512,(3,3),activation='relu', padding='same')(conv5)
    pool5 = MaxPooling2D((2,2))(conv5)

    conv6 = Convolution2D(1024,(3,3),activation='relu', padding='same')(pool5)
    conv6 = Convolution2D(1024,(3,3),activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D((2,2))(conv6),conv5])
    conv7 = Convolution2D(512,(3,3),activation='relu', padding='same')(up7)
    conv7 = Convolution2D(512,(3,3),activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D((2,2))(conv7),conv4])
    conv8 = Convolution2D(256,(3,3),activation='relu', padding='same')(up8)
    conv8 = Convolution2D(256,(3,3),activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D((2,2))(conv8),conv3])
    conv9 = Convolution2D(128,(3,3),activation='relu', padding='same')(up9)
    conv9 = Convolution2D(128,(3,3),activation='relu', padding='same')(conv9)

    up10 = concatenate([UpSampling2D((2,2))(conv9),conv2])
    conv10 = Convolution2D(64,(3,3),activation='relu', padding='same')(up10)
    conv10 = Convolution2D(64,(3,3),activation='relu', padding='same')(conv10)

    up11 = concatenate([UpSampling2D((2,2))(conv10),conv1])
    conv11 = Convolution2D(32,(3,3),activation='relu', padding='same')(up11)
    conv11 = Convolution2D(32,(3,3),activation='relu', padding='same')(conv11)

    conv12 = Convolution2D(1,(3,3),activation='sigmoid')(conv11)
    """
    u_net = Model(input=inputs, output=conv6)
    u_net.summary()
    return u_net


(x_train,_), (x_test,_) = mnist.load_data()

x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
x_train = np.reshape(x_train, (len(x_train),28,28,1))
x_test = np.reshape(x_train, (len(x_train),28,28,1))

noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0,scale=1.0,size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0,scale=1.0,size=x_test.shape)

x_train_noisy = np.clip(x_train_noisy,0.,1.)
x_test_noisy = np.clip(x_test_noisy,0.,1.)

# make model
autoencoder = create_u_net([28,28])
autoencoder.compile(optimizer='adadelta',loss='binary_crossentropy')

autoencoder.fit(
    x_train_noisy,
    x_train,
    epochs=50,
    batch_size=128,
    shuffle=True,
    validation_data=(x_test_noisy,x_test)
)

decoded_imgs = autoencoder.predict(x_test_noisy)

n = 10
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2,n,i+1)
    plt.imshow(x_test_noisy[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2,n,i+1 + n)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()