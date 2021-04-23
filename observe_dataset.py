import numpy as np
import matplotlib.pyplot as plt
from utils import load_dataset, im_convert

# data path
path = 'ants_and_bees'

# load_dataset
training_loader, validation_loader = load_dataset(path)

# classes
classes = ('ant', 'bee')

dataiter = iter(training_loader)
images, labels = dataiter.next()
fig = plt.figure(figsize=(25, 4))

for idx in np.arange(20):
  ax = fig.add_subplot(2, 10, idx+1, xticks=[], yticks=[])
  plt.imshow(im_convert(images[idx]))
  ax.set_title(classes[labels[idx].item()])

plt.show()