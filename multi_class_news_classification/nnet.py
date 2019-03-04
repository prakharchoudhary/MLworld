from keras import models
from keras import layers
from keras import optimizers
from keras.models import model_from_json
import pickle


class Model:
    """A simple neural network to classify the newswires, as per multiple labels"""

    def __init__(self, data, labels, lr=0.001):
        """
        - Initialize all the variable to be used by the Neural Network.
        - Split the data and labels into training and validation set.
        """
        self.train_data = data[1000:]
        self.train_labels = labels[1000:]
        self.val_data = data[:1000]
        self.val_labels = labels[:1000]
        self.model = models.Sequential()
        self.lr = lr
        self.history = None

    def network(self):
        """
        A three layer Neural Network.
        - Input layer : 64 nodes, ReLU activation
        - Hidden layer : 64 nodes, ReLU activation
        - Output layer : 64 nodes, 'softmax' activation
        - Optimizer : RMSprop
        - Loss function : Categorical Cross-entropy
        - Metrics : Accuracy
        """
        self.model.add(layers.Dense(
            64, activation='relu', input_shape=(10000,)))
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(46, activation='softmax'))
        self.model.compile(optimizer=optimizers.RMSprop(self.lr),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def train_model(self):
        """Train the model and store the details."""
        self.history = self.model.fit(self.train_data,
                                      self.train_labels,
                                      epochs=20,
                                      batch_size=512,
                                      validation_data=(self.val_data, self.val_labels))
        self.save_model()
        with open("history.pkl", "wb") as file:
            pickle.dump(self.history.history, file)

    def save_model(self):
        """Save the model architecture in a json file and store weights in h5 format."""
        model_json = self.model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("model.h5")
        print("Saved model to disk")

    def load_model(self):
        """Load the model to use for prediction on new data."""
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("model.h5")
        self.model = loaded_model
        print("Loaded model from disk")
