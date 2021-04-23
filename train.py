import torch
import numpy as np

from torch import nn
from torchvision import models
from utils import load_dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# data path
path = 'ants_and_bees'

# load_dataset
training_loader, validation_loader = load_dataset(path)

# classes
classes = ('ant', 'bee')

# load model
#model = models.alexnet('pretrained')
model = models.vgg16('pretrained')

# model modification
for params in model.features.parameters():
    params.requires_grad = False

n_inputs = model.classifier[6].in_features
last_layer = nn.Linear(n_inputs, len(classes))
model.classifier[6] = last_layer

model.to(device)
#print(model.classifier[6].out_features)

# loss
criterion = nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)

# training
epochs = 5
running_loss_history = []
running_corrects_history = []
val_running_loss_history = []
val_running_corrects_history = []

for e in range(epochs):

    running_loss = 0.0
    running_corrects = 0.0
    val_running_loss = 0.0
    val_running_corrects = 0.0

    for inputs, labels in training_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)

        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data)

    else:
        with torch.no_grad():
            for val_inputs, val_labels in validation_loader:
                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device)
                val_outputs = model(val_inputs)

                val_loss = criterion(val_outputs, val_labels)

                _, val_preds = torch.max(val_outputs, 1)
                val_running_loss += val_loss.item()
                val_running_corrects += torch.sum(val_preds == val_labels.data)

        epoch_loss = running_loss / len(training_loader.dataset)
        epoch_acc = running_corrects.float() / len(training_loader.dataset)
        running_loss_history.append(epoch_loss)
        running_corrects_history.append(epoch_acc)

        val_epoch_loss = val_running_loss / len(validation_loader.dataset)
        val_epoch_acc = val_running_corrects.float() / len(validation_loader.dataset)
        val_running_loss_history.append(val_epoch_loss)
        val_running_corrects_history.append(val_epoch_acc)

        print('Epoch :', (e + 1))
        print('training_loss: {:.4f}, training_acc {:.4f} '.format(epoch_loss, epoch_acc.item()))
        print('val_loss: {:.4f}, val_acc {:.4f} '.format(val_epoch_loss, val_epoch_acc.item()))
        print('\n')

print("Training finished.")

# save history
np.save('running_loss_history.npy', running_loss_history)
np.save('val_running_loss_history.npy', val_running_loss_history)
np.save('running_corrects_history.npy', running_corrects_history)
np.save('val_running_corrects_history.npy', val_running_corrects_history)
print("History Saved...")

# save model
model_path = 'D:/Models/model'
torch.save(model, model_path)
print("Model Saved...")