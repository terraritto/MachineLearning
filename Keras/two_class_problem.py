from keras.datasets import imdb
import numpy as np

def VectorizeSequences(sequences, dimension=10000):
    #make matrix of sharp(len(sequences,dim)) and fill zero
    results = np.zeros(len(sequences,dimension))

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
'''
# How to decode
word_index = imdb.get_word_index()
reverse_word_index = dict((value,key) for (key,value) in word_index.items())
decoded_review = ' '.join([reverse_word_index.get(i-3,'?') for i in train_data[0]])
print(decoded_review)
'''