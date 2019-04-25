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
from easydict import EasyDict as edict
import pandas as pd

from configs import configs


#SIMNAME = 'first_long_train'
PORT = 6065
CONFIG = 'long_train_0'
opts = edict(configs[CONFIG])



""" Delete all figures """
from visdom import Visdom
viz = Visdom(port=PORT)
# for env in viz.get_env_list():
viz.delete_env(opts.train_identifier)

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
optimizer = optim.SGD(model.parameters(), lr=opts.optimizer.lr, momentum=opts.optimizer.momentum)
scheduler = lr_scheduler.StepLR(optimizer, step_size=opts.lr_scheduler.step_size, gamma=opts.lr_scheduler.gamma)
criterion = nn.CrossEntropyLoss()

""" Model training"""
num_epochs = 1000
steps_per_epoch = 40
since = time.time()

best_model_wts = copy.deepcopy(model.state_dict())
best_acc = -float('inf')



plotter  = VisdomLinePlotter(env_name=opts.train_identifier)


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

        running_class_stats = {classname: {'TP':0, 'FP':0, 'TN':0, 'FN':0} for classname in dataset_val.classnames}


        running_class_corrects = {i: 0 for i in range(5)}
        running_class_wrongs = {i: 0 for i in range(5)}

        # Iterate over data once.
        batchidx = 0
        for inputs, labels in tqdm(dataloaders[phase]):
        #for i_step in tqdm(range(steps_per_epoch), desc='step'):
        #    inputs, labels = next(dataloaders[phase])

            # batchidx += 1
            # if batchidx>10:
            #     print(" ")
            #     break

            inputs, labels = inputs.to(device), labels.to(device)

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
                classname = dataset_val.index2name[gt]
                running_class_stats[classname]['TP'] += int(torch.sum( (preds == gt) & (labels == gt)))
                running_class_stats[classname]['TN'] += int(torch.sum( (preds != gt) & (labels != gt)))
                running_class_stats[classname]['FP'] += int(torch.sum( (preds == gt) & (labels != gt)))
                running_class_stats[classname]['FN'] += int(torch.sum( (preds != gt) & (labels == gt)))



        epoch_loss = running_loss / len(dataloaders[phase])
        epoch_acc = float(running_corrects) / (float(running_corrects) + float(running_wrongs) + 1 )

        class_acc = {i: float(running_class_corrects[i]) / (float(running_class_corrects[i]) + float(running_class_wrongs[i]) + 1e-6 ) for i in range(5)}

        """ Plot statistics to visdom """
        plotter.plot('acc', phase, 'acc', epoch, epoch_acc)
        plotter.plot(var_name='loss', split_name=phase, title_name='loss', x=epoch, y=epoch_loss)
        plotter.plot(var_name='LR', split_name='LR', title_name='LR', x=epoch, y=optimizer.param_groups[0]['lr'])
        plotter.plot(var_name='LR', split_name='LR', title_name='LR', x=epoch, y=optimizer.param_groups[0]['lr'])

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
        for key, val in class_acc.items():
            print(key, val)
            classname = dataset_val.index2name[key]
            TP = running_class_stats[classname]['TP']
            TN = running_class_stats[classname]['TN']
            FP = running_class_stats[classname]['FP']
            FN = running_class_stats[classname]['FN']
            class_acc = round(float((TP+TN)/(TP+TN+FP+FN+1e-3)), 2)
            class_prec = round(float(TP/(TP+FP+1e-3)), 2)
            class_recall = round(float(TP/(TP+FN+1e-3)), 2)
            print(TP,TN,FP,FN,'ooo')
            print(class_acc, class_prec, class_recall, "---")

            #plotter_acc.plot(var_name=key, split_name=phase, title_name='class_acc', x=epoch, y=val)
            plotter.plot(var_name='acc_' + phase, split_name=classname, title_name='class_acc', x=epoch,
                             y=class_acc)
            plotter.plot(var_name='prec_' + phase, split_name=classname, title_name='class_prec', x=epoch,
                             y=class_prec)
            plotter.plot(var_name='recall_' + phase, split_name=classname, title_name='class_recall', x=epoch,
                             y=class_recall)

        # deep copy the model
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            # save the currently best model to disk
            torch.save(best_model_wts, './checkpoints/'+opts.train_identifier+'.'+str(epoch))
            print("Saved new best checkpoint to disk")



    print()

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
print('Best val Acc: {:4f}'.format(best_acc))

# load best model weights
model.load_state_dict(best_model_wts)
###return model


