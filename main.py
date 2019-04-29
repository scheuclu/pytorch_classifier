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
ngpus = py3nvml.grab_gpus(num_gpus=1, gpu_fraction=0.95, gpu_select=range(1,8))

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
                    default='debug',
                    help='Name of configuration')

parser.add_argument('--port',
                    type=int,
                    default='6065',
                    help='Port for visdom usage')


args = parser.parse_args()
opts = edict(configs[args.config])


""" Delete all figures """
from visdom import Visdom
viz = Visdom(port=args.port)
# for env in viz.get_env_list():
viz.delete_env(opts.train_identifier)



if opts.cross_val_phase ==0:

    traindir = '/raid/user-data/lscheucher/tmp/pytorch_classifier_data/train0'
    valdir = '/raid/user-data/lscheucher/tmp/pytorch_classifier_data/val0'
else:
    traindir = '/raid/user-data/lscheucher/tmp/pytorch_classifier_data/train1'
    valdir = '/raid/user-data/lscheucher/tmp/pytorch_classifier_data/val1'


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

if False:#TODO
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
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
    model.fc = nn.Linear(512, 5, bias=True)



    # print("Loading")
    # #model = models.squeezenet.SqueezeNet(num_classes=5)
    # #model = models.squeezenet1_1(pretrained=True, num_classes=5)
    #
    # model = models.squeezenet.SqueezeNet(num_classes=5)
    # checkpoint = torch.load('/raid/user-data/lscheucher/tmp/pytorch_classifier_models/squeezenet1_1-f364aa15.pth')
    # model.load_state_dict(checkpoint['model_state_dict'])
    # #
    # # print("Loading")
    # # model = torch.load('/raid/user-data/lscheucher/tmp/pytorch_classifier_models/squeezenet1_1-f364aa15.pth')
    # print("Loaded")
elif opts.model == 'ResNet18':

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, 5, bias=True)
elif opts.model == 'DenseNet169':

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, 5, bias=True)
else:
    raise ValueError("model not supported",opts.model)
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
            run_epoch(phase, model, criterion, optimizer, scheduler, dataloaders[phase], device)

        lr = optimizer.param_groups[0]['lr']
        plot_epoch_end(plotter=plotter, phase=phase, epoch=epoch, epoch_acc=epoch_acc, epoch_loss=epoch_loss,
                       lr=lr, running_class_stats=running_class_stats)

        # deep copy the model
        if phase == 'val':# and epoch_acc > best_acc:
            #best_model_wts = copy.deepcopy(model.state_dict())
            # save the currently best model to disk
            torch.save(model.state_dict(), './checkpoints/'+opts.train_identifier+'.'+str(epoch))
            print("Saved new best checkpoint to disk")

    print()

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
print('Best val Acc: {:4f}'.format(best_acc))

# load best model weights
model.load_state_dict(best_model_wts)
###return model


