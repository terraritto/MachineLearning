from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
from keras import optimizers
import os
import numpy as np
import matplotlib.pyplot as plt


conv_base = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(150,150,3)
)
conv_base.summary()
# freezing: not update weight
conv_base.trainable = True

#-----fine tuning-----
conv_base.trainable = True

set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        layer.trainable = True
    else:
        layer.trainable = False
#----------------------
base_dir = "C:/Users/admin/Desktop/python/machine learning/cats_and_dogs_small"

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir,'test')

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary'
)

# define Dense classification and train
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))
model.summary()

print("weight:{}".format(len(model.trainable_weights)))

model.compile(
    optimizer=optimizers.RMSprop(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['acc']
)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=50
)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

# plot accuracy
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

#plot loss
plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs, val_loss,'b',label='Validation loss')
plt.title('Taining and validation loss')
plt.legend()
plt.show()

# plot smoothing
def smooth_curve(points, factor=0.8):
    smoothed_points=[]
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1-factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

# accuracy
plt.plot(epochs,smooth_curve(acc),'bo',label='Smoothed training acc')
plt.plot(epochs,smooth_curve(val_acc),'b',label='Smoothed validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

# loss
plt.plot(epochs,smooth_curve(loss),'bo', label='Smoothed training loss')
plt.plot(epochs, smooth_curve(val_loss),'b',label='Smoothed validation loss')
plt.title("Training and validation loss")
plt.legend()
plt.show()

#test
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary'
)

test_loss, test_acc = model.evaluate_generator(test_generator,steps=50)
print('test acc:' test_acc)