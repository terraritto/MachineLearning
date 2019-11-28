from keras.datasets import reuters
from keras import models
from keras import layers
from keras.utils.np_utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
# this problem is single-label multiclass classification

def VectorizeSequences(sequences, dimension=10000):
    results = np.zeros((len(sequences),dimension))
    for i, sequence in enumerate(sequences):
        results[i,sequence] = 1
    return results

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

"""
#information about size
print(len(train_data))
print(len(test_data))

print(train_data[10])
"""

"""
#how to decode
word_index = reuters.get_word_index()
reverse_word_index = dict([(value,key) for (key,value) in word_index.items()])

decoded_newswire = ' '.join([reverse_word_index.get(i-3,'?') for i in train_data[0]])

print(decoded_newswire)
"""

### Vectorize
x_train = VectorizeSequences(train_data)
x_test = VectorizeSequences(test_data)

one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)
### construct network
# two class has 16 unit,but this problem has 46 labels.
# it means important information may loss.
# now, we set 64 unit.
model = models.Sequential()
model.add(layers.Dense(64,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(46,activation='softmax'))
# it output 46 labels per input.
# softmax: 46 label has probability distribution
# total of 46 label is 1.

### compile
# if cast using np.array instead of one-hot encoding,
# you should use loss='sparse_categorical_crossentropy'
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])

### set validation dataset
x_val = x_train[:1000]
y_val = one_hot_train_labels[:1000]
partial_x_train = x_train[1000:]
partial_y_train = one_hot_train_labels[1000:]

### train the model
history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=8,
    batch_size=512,
    validation_data=(x_val,y_val)
)

###plot loss
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(loss) + 1)

plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs, val_loss,'b',label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

###plot accuracy
plt.clf()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'b',label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

###last result
results = model.evaluate(x_test,one_hot_test_labels)
print(results)

### predict
# it means output is 46 labels and probability distribution
predictions = model.predict(x_test)
print(predictions[0].shape)
print(np.sum(predictions[0]))
print(np.argmax(predictions[0]))