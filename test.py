import torch
import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, im_convert

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load model
path = 'D:/model'
model = torch.load(path)

# classes
classes = ('ant', 'bee')

# load data
data_path = 'ants_and_bees'
_, validation_loader = load_dataset(data_path)

dataiter = iter(validation_loader)
images, labels = dataiter.next()
images = images.to(device)
labels = labels.to(device)
output = model(images)
_, preds = torch.max(output, 1)

fig = plt.figure(figsize=(25, 4))

for idx in np.arange(20):
  ax = fig.add_subplot(2, 10, idx+1, xticks=[], yticks=[])
  plt.imshow(im_convert(images[idx]))
  ax.set_title("{} ({})".format(str(classes[preds[idx].item()]), str(classes[labels[idx].item()])), color=("green" if preds[idx]==labels[idx] else "red"))

plt.show()