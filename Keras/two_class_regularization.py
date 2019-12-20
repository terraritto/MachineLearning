from keras.datasets import imdb
from keras import models
from keras import layers
from keras import regularizers
import numpy as np
import matplotlib.pyplot as plt

def VectorizeSequences(sequences, dimension=10000):
    #make matrix of sharp(len(sequences,dim)) and fill zero
    results = np.zeros((len(sequences),dimension))

    for i, sequence in enumerate(sequences):
        results[i,sequence] = 1 # set index of results[i] 
    return results

#input data
(train_data, train_labels), (test_data,test_labels) = imdb.load_data(num_words=10000)

# vectorize about train data
x_train = VectorizeSequences(train_data)
# vectorize about test data
x_test = VectorizeSequences(test_data)
# vectorize about labels
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# definition of model

"""
#L2 normalization
model = models.Sequential()
model.add(layers.Dense(16,kernel_regularizer=regularizers.l2(0.001),
activation='relu',input_shape=(10000,)))
model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),
activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))
"""

# Dropout layer
model = models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1,activation='sigmoid'))

#compile the model
# Loss function: 
# crossentropy is the best because this problem output probability.
# now, we use two value, so use binary_crossentropy(二値の交差エントロピー)
# other selection is meah_squared_error(平均二乗誤差)
# Optimizer: rmsprop
model.compile(optimizer='rmsprop', loss='binary_crossentropy',metrics=['acc'])

# make checking data set
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# training model
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val,y_val))

results = model.evaluate(partial_x_train,partial_y_train)
print(results)
# training
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)

# print loss
# 'bo' means 'blue dot'
plt.plot(epochs, loss_values, 'bo', label='Training loss')
# 'b' means 'solid blue line'
plt.plot(epochs,val_loss_values,'b',label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# delete figure
plt.clf()

# print accuracy
acc = history_dict['acc']
val_acc = history_dict['val_acc']
plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# generate Likelihood
print(model.predict(x_test))
'''
# How to decode
word_index = imdb.get_word_index()
reverse_word_index = dict((value,key) for (key,value) in word_index.items())
decoded_review = ' '.join([reverse_word_index.get(i-3,'?') for i in train_data[0]])
print(decoded_review)
'''