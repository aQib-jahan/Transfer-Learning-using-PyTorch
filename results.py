import matplotlib.pyplot as plt
import numpy as np

# load history
running_loss_history = np.load('running_loss_history.npy')
val_running_loss_history = np.load('val_running_loss_history.npy')
running_corrects_history = np.load('running_corrects_history.npy', allow_pickle=True)
val_running_corrects_history = np.load('val_running_corrects_history.npy', allow_pickle=True)

plt.plot(running_loss_history, label='training loss')
plt.plot(val_running_loss_history, label='validation loss')
plt.legend()
plt.show()

plt.plot(running_corrects_history, label='training accuracy')
plt.plot(val_running_corrects_history, label='validation accuracy')
plt.legend()
plt.show()