import numpy as np
from keras.datasets import reuters
import nnet
import pickle

# load the dataset and prepare train and test data
(train_data, train_labels), (test_data, test_labels) = \
    reuters.load_data(num_words=10000)

# decoding newswires back to text
word_index = reuters.get_word_index()
reverse_word_index = dict([(value, word)
                           for word, value in word_index.items()])
decoded_newswire = ' '.join(
    [reverse_word_index.get(i - 3, '?') for i in train_data[0]])


# vectorizing data
def vectorize_sequences(sequences, dimensions=10000):
    results = np.zeros((len(sequences), dimensions))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

X_train = vectorize_sequences(train_data)
X_test = vectorize_sequences(test_data)

# test to check encoding of data
assert len(X_train[0]) == 10000


# One-hot encoding the labels
def one_hot_encoding(labels, dims=46):
    results = np.zeros((len(labels), dims))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results

one_hot_train_labels = one_hot_encoding(train_labels)
one_hot_test_labels = one_hot_encoding(test_labels)

# test to check encoding of labels
assert len(one_hot_train_labels[0]) == 46


# Train the model
network = nnet.Model(X_train, one_hot_train_labels)
network.network()
network.train_model()

# evaluate and print results
results = network.model.evaluate(X_test, one_hot_test_labels)
print("The results are: ", str(results))
