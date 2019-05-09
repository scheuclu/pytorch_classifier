from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from plots import VisdomLinePlotter, plot_epoch_end
from train import run_epoch
import time
import copy
import py3nvml
ngpus = py3nvml.grab_gpus(num_gpus=1, gpu_fraction=0.95, gpu_select=range(0,8))

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
plt.ion()   # interactive mode
from easydict import EasyDict as edict

from configs import configs
pickle_file = '/raid/user-data/lscheucher/projects/bounding_box_classifier/full_object_index.pickle'

from collections import namedtuple
IndexEntry = namedtuple('IndexEntry', ['img_path', 'sub_idx', 'classname', 'left', 'top', 'right', 'bottom'], verbose=False)


import argparse

parser = argparse.ArgumentParser(description='Train pytorch classifier.')

parser.add_argument('--config',
                    type=str,
                    default='squeezenet_0_other_random',
                    help='Name of configuration')

parser.add_argument('--port',
                    type=int,
                    default='6065',
                    help='Port for visdom usage')


# parser.add_argument('--imageset',
#                     type=str,
#                     default='imageset_5_other_random', #imageset_5, imageset_5_other, imageset_5_other_random
#                     help='TODO')

# parser.add_argument('--numclasses',
#                     type=int,
#                     default='6',
#                     help='Number of classes that the network can differentiate')



args = parser.parse_args()
#opts = edict(configs[args.config])
opts = configs[args.config]

import os
#if not os.path.isdir()





""" Define appropraite directories for train and validation images"""
if opts.cross_val_phase ==0:

    traindir = os.path.join('/raid/user-data/lscheucher/tmp/pytorch_classifier_data',opts.imageset,'train0')
    valdir = os.path.join('/raid/user-data/lscheucher/tmp/pytorch_classifier_data',opts.imageset,'val0')
else:
    traindir = os.path.join('/raid/user-data/lscheucher/tmp/pytorch_classifier_data',opts.imageset,'train1')
    valdir = os.path.join('/raid/user-data/lscheucher/tmp/pytorch_classifier_data',opts.imageset,'val1')


""" Define directory for output """
#This directory will be used  both for evaluation and for saving the plots
if not os.path.isdir(os.path.join('/raid/user-data/lscheucher/tmp/pytorch_classifier_checkpoints', opts.imageset)):
    os.mkdir(os.path.join('/raid/user-data/lscheucher/tmp/pytorch_classifier_checkpoints', opts.imageset))
outdir = os.path.join('/raid/user-data/lscheucher/tmp/pytorch_classifier_checkpoints', opts.imageset, opts.train_identifier)
if not os.path.isdir(outdir):
    os.mkdir(outdir)


""" Delete all figures """
from visdom import Visdom
visdom_log_path = os.path.join(outdir,opts.train_identifier+".visdom")
#visdom_log_path = outdir
print("Saving visdom logs to", visdom_log_path)
viz = Visdom(port=args.port, log_to_filename=visdom_log_path)
# for env in viz.get_env_list():
viz.delete_env(opts.train_identifier)
viz.log_to_filename = os.path.join(outdir,opts.train_identifier+".visdom")


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        normalize,
    ]))

import math
if False:#TODO WeightedRandomSampler not working
    class_sample_count = [len([i for i in train_dataset.imgs if i[1]==classid]) for classid in range(len(train_dataset.classes))] # dataset has 10 class-1 samples, 1 class-2 samples, etc.
    #class_sample_count = [math.sqrt(i) / min(class_sample_count) for i in class_sample_count]
    weights = 1 / torch.Tensor(class_sample_count)
    print(len(train_dataset),"len(train_dataset)")
    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, int(len(train_dataset)/opts.batchsize)-1, replacement=True)
else:
    train_sampler = None


train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=opts.batchsize, shuffle=(train_sampler is None),
    num_workers=20, pin_memory=True, sampler=train_sampler)

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=opts.batchsize, shuffle=False,
    num_workers=20, pin_memory=True)


dataloaders={

    'train': train_loader,
    'val': val_loader
}

""" Model, Optimizer, Scheduler"""
if opts.model == 'SqueezeNet':
    model = models.squeezenet1_0(pretrained=True)
    model.fc = nn.Linear(512, len(train_dataset.classes), bias=True)
elif opts.model == 'ResNet18':

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, len(train_dataset.classes), bias=True)
elif opts.model == 'ResNet152':

    model = models.resnet18(pretrained=True)
    Linear(in_features=2048, out_features=len(train_dataset.classes), bias=True)
elif opts.model == 'DenseNet169':

    model = models.densenet169(pretrained=True)
    model.classifier = nn.Linear(1664, len(train_dataset.classes), bias=True)

elif opts.model == 'DenseNet201':
    model = models.densenet201(pretrained=True)
    model.classifier = nn.Linear(1920, len(train_dataset.classes), bias=True)
else:
    raise ValueError("model not supported", opts.model)

#device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
device = 'cuda:0'
model = model.to(device)

""" optimizer """
optimizer = optim.SGD(model.parameters(), lr=opts.optimizer.lr, momentum=opts.optimizer.momentum)
scheduler = lr_scheduler.StepLR(optimizer, step_size=opts.lr_scheduler.step_size, gamma=opts.lr_scheduler.gamma)
criterion = nn.CrossEntropyLoss()

print("Visdom env name:",opts.train_identifier)
plotter  = VisdomLinePlotter(env_name=opts.train_identifier, plot_path=outdir)


""" Model training """
num_epochs = 100
steps_per_epoch = 40
since = time.time()

best_model_wts = copy.deepcopy(model.state_dict())
best_acc = -float('inf')

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    for phase in ['train', 'val']:
        running_class_stats, epoch_loss, epoch_acc, class_acc =\
            run_epoch(phase, model, criterion, optimizer, scheduler, dataloaders[phase], device)

        lr = optimizer.param_groups[0]['lr']
        plot_epoch_end(plotter=plotter, phase=phase, epoch=epoch, epoch_acc=epoch_acc, epoch_loss=epoch_loss,
                       lr=lr, running_class_stats=running_class_stats)
        plotter.save_plots()

        # deep copy the model
        if phase == 'val':# and epoch_acc > best_acc:
            #best_model_wts = copy.deepcopy(model.state_dict())
            # save the currently best model to disk
            torch.save(model.state_dict(), outdir+'/'+str(epoch)+'.pth')
            print("Saved new best checkpoint to:\n"+outdir+'/'+str(epoch)+'.pth')

    print()

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
print('Best val Acc: {:4f}'.format(best_acc))

# load best model weights
model.load_state_dict(best_model_wts)
###return model


