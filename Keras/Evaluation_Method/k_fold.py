
k = 4
num_validation_samples = len(data) // k
np.random.shuffle(data)
validation_scores = []

for fold in range(k):
    # select validation data
    validation_data = data[num_validation_samples * fold:num_validation_samples*(fold+1)]

    # other data is training data
    training_data = data[:num_validation_samples*fold] +
    data[num_validation_samples*(fold+1):]

    # make new instance of model
    model = get_model()
    # train model
    model.train(training_data)

    # add validation score
    validation_score = model.evaluate(validation_data)
    validation_scores.append(validation_score)

# validation score is average for k-fold
validation_score = np.average(validation_scores)

# we train the finally model with data which don't use test
model = get_model()
model.train(data)
test_score = model.evaluate(test_data)