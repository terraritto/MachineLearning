import numpy as np

num_validation_samples = 10000

# normally, data is shuffled.
np.random.shuffle(data)

# define Validation data
validation_data = data[:num_validation_samples]
data = data[num_validation_samples:]
# define training data
training_data = data[:]

# we train the model with training data and evaluate validation data
model = get_model()
model.train(training_data)
validation_score = model.evaluate(validation_data)

# this time, we can loop model of turning , re-training, evaluate, re-training ......

# if end tuning about hyper parameter(it means layer dense etc),
# we train the finally model with data which don't use test
model = get_model()
model.train(np.concatenate([training_data,validation_data]))
test_score = model.evaluate(test_data)