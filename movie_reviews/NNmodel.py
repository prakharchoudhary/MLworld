from keras import models
from keras import layers
from keras import optimizers
from matplotlib import pyplot as plt
import pickle


class NeuralNet:

    def __init__(self, x_train, y_train, lr=0.001):
        self.model = models.Sequential()
        self.x_train = x_train
        self.y_train = y_train
        self.partial_x_train = None
        self.partial_y_train = None
        self.x_val = None
        self.y_val = None
        self.lr = lr
        self.history = None

    def divide_data(self):
        self.x_val = self.x_train[:10000]
        self.y_val = self.y_train[:10000]
        self.partial_x_train = self.x_train[10000:]
        self.partial_y_train = self.y_train[10000:]

    def network(self):
        self.model.add(layers.Dense(
            16, activation='relu', input_shape=(10000,)))
        self.model.add(layers.Dense(16, activation='relu'))
        self.model.add(layers.Dense(1, activation='sigmoid'))
        self.model.compile(optimizer=optimizers.RMSprop(self.lr),
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

    def train_model(self):
        self.history = self.model.fit(self.partial_x_train,
                                      self.partial_y_train,
                                      epochs=20,
                                      batch_size=512,
                                      validation_data=(self.x_val, self.y_val))
        self.save_model()
        with open("history.pkl", "wb") as file:
            pickle.dump(self.history.history, file)

    def save_model(self):
        model_json = self.model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("model.h5")
        print("Saved model to disk")

    def load_model(self):
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("model.h5")
        print("Loaded model from disk")
