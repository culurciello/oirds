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
import os, sys

plt.ion()   # interactive mode 

print('Usage: python3 test.py modelDef.pth finemodel.pth')


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.axis('off')
    plt.pause(0.001)  # pause a bit so that plots are updated

    return plt


# params:
batch_size = 16
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

# loop many examples:
while True:
    # Get a batch of training data
    inputs, classes = next(iter(dset_loaders['val']))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)
    fig = imshow(out)

    #load model
    model = torch.load(sys.argv[1])
    model.load_state_dict(torch.load(sys.argv[2]))
    model.cpu()
    model.eval()
    # wrap them in Variable
    inputs = Variable(inputs)

    # forward
    outputs = model(inputs)
    _, preds = torch.max(outputs.data, 1)

    txt=''
    for i in range(len(preds)):
        if i%8==0: 
            txt = txt+'\n'
        if preds[i][0]==0:
            txt = txt+'car, '
        else:
            txt = txt+'---, '

    print(txt)
    fig.suptitle(txt, fontsize=15)
    plt.show()
    input()




