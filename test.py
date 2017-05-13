#!/usr/bin/env python

# Eugenio Culurciello, May 2017
# test on OIRDS dataset
# updated code from: http://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

plt.ion()   # interactive mode 


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# params:
batch_size = 128
threads = 8

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
     transforms.RandomSizedCrop(224),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
     transforms.Scale(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'data/oirds/'
dsets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
         for x in ['train', 'val']}
dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size,
                                               shuffle=True, num_workers=threads)
                for x in ['train', 'val']}
dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
dset_classes = dsets['train'].classes

use_gpu = torch.cuda.is_available()


# Get a batch of training data
inputs, classes = next(iter(dset_loaders['val']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)
imshow(out, title=[dset_classes[x] for x in classes])

#load model
model = torch.load("modelDef.pth") #models.resnet18()
model.load_state_dict(torch.load("finemodel.pth"))
model.eval()
# wrap them in Variable
if use_gpu:
    inputs = Variable(inputs.cuda())


# forward
outputs = model(inputs)
_, preds = torch.max(outputs.data, 1)

print(preds)



