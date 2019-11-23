from keras.datasets import reuters
# this problem is single-label multiclass classification

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)