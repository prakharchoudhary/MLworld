from keras.datasets import imdb
import matplotlib.pyplot as plt
import NNmodel
import numpy as np

(train_data, train_labels), (test_data,
                             test_labels) = imdb.load_data(num_words=10000)


def vectorize_data(sequences, dimensions=10000):
    """ Vectorize the array of integers per document """
    results = np.zeros((len(sequences), dimensions))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

# Preparing the train and test feature set
X_train = vectorize_data(train_data)
X_test = vectorize_data(test_data)

# Preparing the train and test labels
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

network = NNmodel.NeuralNet(X_train, y_train)
network.divide_data()
network.network()
network.train_model()
