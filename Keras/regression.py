from keras.datasets import boston_housing
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

### definition of model
# point: train data is small = model may be over-training
# to avoid it, you should be use small network.
# activation: isn't set. it means a(x)=x ,linear
# loss: mean squared error(平均二乗誤差), often use regression
# metrics: mean absolute error(平均絶対誤差)
def BuildModel():
    model = models.Sequential()
    model.add(layers.Dense(64,activation='relu',input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
    return model

### smooth point for EMA
def smooth_curve(points,factor=0.9):
    smoothed_curve = []
    for point in points:
        if smoothed_curve:
            previous = smoothed_curve[-1]
            smoothed_curve.append(previous*factor + point*(1-factor))
        else:
            smoothed_curve.append(point)
    return smoothed_curve

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

### information about size
"""
print("traindata: {}".format(train_data.shape))
print("testdata: {}".format(test_data.shape))
print("train_targets: {}".format(train_targets.shape))
"""

### normalize data
# first mean feature value(column of input matrix) and sub
mean = train_data.mean(axis=0)
train_data -= mean
# second, div standard deviation.
# finally,center of feature value = 0, standard deviation = 0
std = train_data.std(axis=0)
train_data /= std
# test data use train data's mean and std.
test_data -= mean
test_data /= std

### tuning section
### k-fold cross-validation
"""
k = 4
num_val_samples = len(train_data) // k
num_epochs = 500
all_mae_histories = []

for i in range(k):
    print('processing fold #', i)
    # prepare validation data : data of fold i
    val_data = train_data[i*num_val_samples: (i+1)*num_val_samples]
    val_targets = train_targets[i*num_val_samples: (i+1)*num_val_samples]

    # prepare training data : data of other fold
    partial_train_data = np.concatenate(
        [train_data[:i*num_val_samples],
        train_data[(i+1)*num_val_samples:]],
        axis=0)

    partial_train_targets = np.concatenate(
        [train_targets[:i*num_val_samples],
        train_targets[(i+1)*num_val_samples:]],
        axis=0)
    
    #construct model
    model = BuildModel()

    # conform model in silent mode(verbose=0)
    history = model.fit(
        partial_train_data, partial_train_targets,
        validation_data=(val_data,val_targets),
        epochs=num_epochs,batch_size=1,verbose=0)
    
    mae_history = history.history['val_mae']
    all_mae_histories.append(mae_history)

### construct mean score of k-fold
average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

### plot validation score
plt.plot(range(1,len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

### plot validation score removing first 10 point
smooth_mae_history = smooth_curve(average_mae_history[10:])
plt.plot(range(1,len(smooth_mae_history)+1),smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()
"""

### final result
model = BuildModel()
model.fit(train_data,train_targets,epochs=80,batch_size=16,verbose=0)
test_mse_score,test_mae_score = model.evaluate(test_data,test_targets)

print(test_mae_score)