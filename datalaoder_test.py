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
                                             num_workers=30)
dataset_loader_val = torch.utils.data.DataLoader(dataset_val,
                                             batch_size=100, shuffle=True,
                                             num_workers=30)

dataloaders={
    'train': iter(dataset_loader_train),
    'val': iter(dataset_loader_val)
}


start= time.time()
for i in range(10):
    inputs, labels = next(dataloaders['train'])
end=time.time()

print(end-start)


"""
with resize
8: 18.397505521774292
6: 25.312967777252197
4: 32.52280306816101
2: 46.636823415756226


without resize
8:
6:
4:
2:
"""