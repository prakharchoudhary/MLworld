import pickle
from matplotlib import pyplot as plt

file = open("history.pkl", 'rb')
history_dict = pickle.load(file)
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, 21)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values,
         'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("loss_graph.png", dpi=150)

# plot the training and validation accuracy
plt.clf()
loss_values = history_dict['acc']
val_loss_values = history_dict['val_acc']
plt.plot(epochs, loss_values, 'bo',
         label='Training Accuracy')
plt.plot(epochs, val_loss_values, 'b',
         label='Validation Accuracy')
plt.title('Training and validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("acc_graph.png", dpi=150)
