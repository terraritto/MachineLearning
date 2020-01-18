from keras.applications import VGG16
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

model = VGG16(weights='imagenet',include_top=False)

# utility function convert valid image
def deprocess_image(x):
    # normalize tensol: center 0, std 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clipping [0,1]
    x += 0.5
    x = np.clip(x,0,1)

    # convert RGB array
    x *= 255
    x = np.clip(x,0,255).astype('uint8')
    return x

# function visualize filter
def generate_pattern(layer_name, filter_index, size=150):
    #construct loss function that maximize filter activation for n-th targe layer
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:,:,:,filter_index])

    # calculate gradienct of input image
    grads = K.gradients(loss, model.input)[0]

    # normalize grad
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # fuction that return loss value and gradient as input image
    iterate = K.function([model.input], [loss, grads])

    # use image including noise
    input_img_data = np.random.random((1,size,size,3)) * 20 + 128

    # exe 40 steps
    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

    img = input_img_data[0]
    return deprocess_image(img)

layer_name = 'block3_conv1'
filter_index = 0

layers = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1']

for layer_name in layers:
    size = 64
    margin = 5
    results = np.zeros((8*size + 7 * margin, 8 * size + 7 * margin,3))

    for i in range(8):
        for j in range(8):
            # generate filete of layer_name
            filter_img = generate_pattern(layer_name,i+(j*8),size=size)

            # arrange result in grid(i,j)
            horizontal_start = i * size + i * margin
            horizontal_end = horizontal_start + size
            vertical_start = j * size + j * margin
            vertical_end  = vertical_start + size
            results[horizontal_start:horizontal_end, vertical_start:vertical_end, :] = filter_img

    # show results grid
    plt.figure(figsize=(20,20))
    plt.imshow(results)
    plt.show()