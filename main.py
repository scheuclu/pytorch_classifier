import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm

from mydataset import MyDataset
from plots import VisdomLinePlotter

import time
import copy

import py3nvml
ngpus = py3nvml.grab_gpus(num_gpus=1, gpu_fraction=0.95, gpu_select=range(1,8))

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode
import pickle
import pandas as pd


SIMNAME = 'test1'


import torch.multiprocessing as multiprocessing
from torch._C import _set_worker_signal_handlers, _update_worker_pids, \
    _remove_worker_pids, _error_if_any_worker_fails
#from . import SequentialSampler, RandomSampler, BatchSampler


pickle_file = '/raid/user-data/lscheucher/projects/bounding_box_classifier/full_object_index.pickle'

from collections import namedtuple
IndexEntry = namedtuple('IndexEntry', ['img_path', 'sub_idx', 'classname', 'left', 'top', 'right', 'bottom'], verbose=False)


""" Load dataset """

with open(pickle_file, 'rb') as f:
    test = pickle.load(f)

dataset_train = MyDataset(pickle_file=pickle_file, mode='train')
dataset_val   = MyDataset(pickle_file=pickle_file, mode='val')


""" Create dataloaders"""
dataset_loader_train = torch.utils.data.DataLoader(dataset_train,
                                             batch_size=100, shuffle=True,
                                             num_workers=40)
dataset_loader_val = torch.utils.data.DataLoader(dataset_val,
                                             batch_size=100, shuffle=True,
                                             num_workers=40)

dataloaders={
    'train': dataset_loader_train,
    'val': dataset_loader_val
}

""" Model """
model = models.squeezenet.SqueezeNet(num_classes=5)
#device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
device = 'cuda:0'
model = model.to(device)

""" optimizer """
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
criterion = nn.CrossEntropyLoss()

""" Model training"""
num_epochs = 1000
steps_per_epoch = 40
since = time.time()

best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0



plotter_acc  = VisdomLinePlotter(env_name='acc')
plotter_loss = VisdomLinePlotter(env_name='loss')


batch_idx_train = 0
batch_idx_val = 0
#start=time.time()
for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    # Each epoch has a training and validation phase

    for phase in ['train', 'val']:
        print(phase.upper())
        if phase == 'train':
            scheduler.step()
            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0
        running_wrongs = 0

        running_class_corrects = {i: 0 for i in range(5)}
        running_class_wrongs = {i: 0 for i in range(5)}

        # Iterate over data once.
        for inputs, labels in tqdm(dataloaders[phase]):


        #for i_step in tqdm(range(steps_per_epoch), desc='step'):
        #    inputs, labels = next(dataloaders[phase])

            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            running_wrongs += torch.sum(preds != labels.data)
            #extended statistics
            for gt in range(5):
                gt_preds = preds[labels == gt]
                running_class_corrects[gt] += torch.sum(gt_preds == gt)
                running_class_wrongs[gt] += torch.sum(gt_preds != gt)


        epoch_loss = running_loss / len(dataloaders[phase])
        epoch_acc = float(running_corrects) / (float(running_corrects) + float(running_wrongs) + 1 )

        class_acc = {i: float(running_class_corrects[i]) / (float(running_class_corrects[i]) + float(running_class_wrongs[i]) + 1e-6 ) for i in range(5)}

        if phase == 'train':
            plotter_acc.plot('acc', 'train', 'acc', epoch, epoch_acc)
            plotter_loss.plot('loss', 'train', 'loss', epoch, epoch_loss)
        else:
            plotter_acc.plot('acc', 'val', 'acc', epoch, epoch_acc)
            plotter_loss.plot('loss', 'val', 'loss', epoch, epoch_loss)

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            phase, epoch_loss, epoch_acc))
        for key, val in class_acc.items():
            print(key, val)


        #reshuffle dataset

        # deep copy the model
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            # save the currently best model to disk
            torch.save(best_model_wts, './checkpoints')
            print("Saved new best checkpoint to disk")



    print()

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
print('Best val Acc: {:4f}'.format(best_acc))

# load best model weights
model.load_state_dict(best_model_wts)
###return model


