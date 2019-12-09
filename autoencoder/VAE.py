import keras
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Lambda, Conv2DTranspose, Convolution2D, MaxPooling2D, UpSampling2D
from keras import backend as K
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

latent_dim = 2 # dimensionality of the latent space
batch_size = 256

decoder = []

def create_VAE(input):
    inputs = Input((input[0],input[1],1))

    conv = Convolution2D(32,(3,3),activation='relu', padding='same')(inputs)
    conv = Convolution2D(64,(3,3),activation='relu', padding='same')(conv)
    conv = Convolution2D(64,(3,3),activation='relu', padding='same')(conv)
    conv = Convolution2D(64,(3,3),activation='relu', padding='same')(conv)

    shape_before = K.int_shape(conv)

    conv = Flatten()(conv)
    conv = Dense(32,activation='relu')(conv)

    z_mean = Dense(latent_dim)(conv)
    z_log_var = Dense(latent_dim)(conv)

    # Latent value Follow N(0,1)
    # use reparameterization trick
    # z = myu + sigma * N(0,1)
    def sampling(args):
        z_mean, z_sigma = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0],latent_dim), mean=0.,stddev=1.)
        return z_mean + K.exp(z_log_var) * epsilon

    #Latent value
    z = Lambda(sampling)([z_mean,z_log_var])

    #input about feeding 'z'
    decode_input = Input(K.int_shape(z)[1:])

    # upsample
    conv_af = Dense(np.prod(shape_before[1:]),activation='relu')(decode_input)

    #Reshape into an image as before 'Flatten' layer
    conv_af = Reshape(shape_before[1:])(conv_af)
    conv_af = Conv2DTranspose(32,(3,3),activation='relu', padding='same')(conv_af)
    conv_af = Convolution2D(1,(3,3),padding='same',activation='sigmoid')(conv_af)

    # concatenate
    VAE = Model(decode_input, conv_af)
    decoder.append(VAE) #add decoder
    VAE.summary()
    z_decode = VAE(z)

    # custom layer for loss function
    class CustomVariationalLayer(keras.layers.Layer):
        def vae_loss(self,x,z_decoded):
            x = K.flatten(x)
            z_decoded = K.flatten(z_decoded)
            xent_loss = keras.metrics.binary_crossentropy(x,z_decoded)
            kl_loss = -5e-4 * K.mean(1+z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=1)
            return K.mean(xent_loss + kl_loss)
        
        def call(self, inputs):
            x = inputs[0]
            z_decode = inputs[1]
            loss = self.vae_loss(x,z_decode)
            self.add_loss(loss,inputs=inputs)
            return x
    
    new_layer = CustomVariationalLayer()([inputs,z_decode])
    
    # network
    vae = Model(inputs, new_layer)
    return vae



(x_train,_), (x_test,_) = mnist.load_data()

x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
x_train = np.reshape(x_train, (len(x_train),28,28,1))
x_test = np.reshape(x_train, (len(x_train),28,28,1))


# make model
autoencoder = create_VAE([28,28])
autoencoder.compile(optimizer='rmsprop',loss=None)
autoencoder.summary()

history = autoencoder.fit(
    x=x_train,
    y=None,
    epochs=10,
    batch_size=128,
    shuffle=True,
    validation_data=(x_test,None)
)

#latent value
n = 15
digit_size = 28
figure = np.zeros((digit_size*n,digit_size*n))

grid_x = norm.ppf(np.linspace(0.05,0.95,n))
grid_y = norm.ppf(np.linspace(0.05,0.95,n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
        x_decoded = decoder[0].predict(z_sample, batch_size=batch_size)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()

# loss
"""
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(range(1,epochs), loss[1:], marker='.', label='loss')
plt.plot(range(1,epochs), val_loss[1:], marker='.', label='val_loss')
plt.legend(loc='best', fontsize=10)
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
"""