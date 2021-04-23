import torch
import numpy as np
from torchvision import datasets, transforms

def load_dataset(path):

    # augmentation
    transform_train = transforms.Compose([transforms.Resize((224, 224)),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomRotation(10),
                                          transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                                          transforms.ColorJitter(brightness=1, contrast=1, saturation=1),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                          ])

    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])

    training_dataset = datasets.ImageFolder(path + '/train', transform=transform_train)
    validation_dataset = datasets.ImageFolder(path + '/val', transform=transform)

    training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=20, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=20, shuffle=False)

    return training_loader, validation_loader

def im_convert(tensor):
  image = tensor.cpu().clone().detach().numpy()
  image = image.transpose(1, 2, 0)
  image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
  image = image.clip(0, 1)

  return image