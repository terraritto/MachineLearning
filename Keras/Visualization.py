from keras.models import load_model
from keras.preprocessing import image
from keras import models
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as pylt

model = load_model('cats_and_dogs_small_1.h5')
model.summary()

img_path = "C:/Users/admin/Desktop/python/machine learning/cats_and_dogs_small/test/cats/cat.1700.jpg"

# preprocessing this image for 4-dim tensor
img = image.load_img(img_path,target_size=(150,150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)

img_tensor /= 255.

# shape=(1,150,150,3)
print(img_tensor.shape)

# show test data
pylt.imshow(img_tensor[0])
pylt.show()

# extract output from output layers
layer_outputs = [layer.output for layer in model.layers[:8]]

# make model that return output for specifical input
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

# return list of five numpy array
activations = activation_model.predict(img_tensor)

first_layer_activation = activations[0]
print(first_layer_activation.shape)

# visualize third channel
pylt.matshow(first_layer_activation[0,:,:,3],cmap='viridis')
pylt.show()

# visualize thirty-th channel
pylt.matshow(first_layer_activation[0,:,:,30], cmap='viridis')
pylt.show()

# visualize all channel
layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)

image_per_row = 15

# show feature map
for layer_name, layer_activation in zip(layer_names,activations):
    # feature num included feature map
    n_features = layer_activation.shape[-1]

    # feature shape(1,size,size,n_features)
    size = layer_activation.shape[1]

    # show tile
    n_cols = n_features // image_per_row
    display_grid = np.zeros((size*n_cols, image_per_row*size))
    
    for col in range(n_cols):
        for row in range(image_per_row):
            channel_image=layer_activation[0,:,:,col*image_per_row+row]

            # after processing
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col*size:(col+1)*size, row*size:(row+1)*size] = channel_image
        
    # show grid
    scale = 1./size
    pylt.figure(figsize=(scale*display_grid.shape[1],scale*display_grid.shape[0]))
    pylt.title(layer_name)
    pylt.grid(False)
    pylt.imshow(display_grid,aspect='auto',cmap='viridis')

pylt.show()