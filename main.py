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
from plots import VisdomLinePlotter, plot_epoch_end
from train import run_epoch
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
pickle_file = '/raid/user-data/lscheucher/projects/bounding_box_classifier/full_object_index.pickle'

from collections import namedtuple
IndexEntry = namedtuple('IndexEntry', ['img_path', 'sub_idx', 'classname', 'left', 'top', 'right', 'bottom'], verbose=False)


""" Parse argv --------------------------------------------------------------"""
import argparse

parser = argparse.ArgumentParser(description='Train pytorch classifier.')

parser.add_argument('--config',
                    type=str,
                    default='long_train_1',
                    help='Name of configuration')

parser.add_argument('--port',
                    type=int,
                    default='6065',
                    help='Port for visdom usage')

args = parser.parse_args()
opts = edict(configs[args.config])
""" -------------------------------------------------------------------------"""


""" -------------------------------------------------------------------------"""
""" Delete all figures """
from visdom import Visdom
viz = Visdom(port=args.port)
# for env in viz.get_env_list():
viz.delete_env(opts.train_identifier)
""" -------------------------------------------------------------------------"""


""" ------------------------------------------------------------------------- """
""" Load dataset """
with open(pickle_file, 'rb') as f:
    test = pickle.load(f)

dataset_train = MyDataset(pickle_file=pickle_file, mode='train')
dataset_val   = MyDataset(pickle_file=pickle_file, mode='val')
""" -------------------------------------------------------------------------"""


""" ------------------------------------------------------------------------- """
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

dataset_val.classnames
""" ------------------------------------------------------------------------- """


""" ------------------------------------------------------------------------- """
""" Model, Optimizer, Scheduler"""
model = models.squeezenet.SqueezeNet(num_classes=5)
#device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
device = 'cuda:0'
model = model.to(device)

""" optimizer """
optimizer = optim.SGD(model.parameters(), lr=opts.optimizer.lr, momentum=opts.optimizer.momentum)
scheduler = lr_scheduler.StepLR(optimizer, step_size=opts.lr_scheduler.step_size, gamma=opts.lr_scheduler.gamma)
criterion = nn.CrossEntropyLoss()

plotter  = VisdomLinePlotter(env_name=opts.train_identifier)
""" ------------------------------------------------------------------------- """


""" ------------------------------------------------------------------------- """
""" Model training """
num_epochs = 1000
steps_per_epoch = 40
since = time.time()

best_model_wts = copy.deepcopy(model.state_dict())
best_acc = -float('inf')

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    for phase in ['train', 'val']:
        running_class_stats, epoch_loss, epoch_acc, class_acc =\
            run_epoch(phase, model, criterion, optimizer, scheduler, dataloaders[phase], device, dataset_val.classnames, dataset_val.index2name)

        lr = optimizer.param_groups[0]['lr']
        plot_epoch_end(plotter=plotter, phase=phase, epoch=epoch, epoch_acc=epoch_acc, epoch_loss=epoch_loss,
                       lr=lr, running_class_stats=running_class_stats, index2name=dataset_val.index2name)

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


