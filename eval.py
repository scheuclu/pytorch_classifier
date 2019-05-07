
import torch
from torchvision import models
#from deprecated.mydataset import MyDataset
import pandas as pd

import py3nvml
ngpus = py3nvml.grab_gpus(num_gpus=1, gpu_fraction=0.95, gpu_select=range(0,8))
device = 'cuda:0'

pickle_file = '/raid/user-data/lscheucher/projects/bounding_box_classifier/full_object_index.pickle'
from collections import namedtuple
IndexEntry = namedtuple('IndexEntry', ['img_path', 'sub_idx', 'classname', 'left', 'top', 'right', 'bottom'], verbose=False)
from easydict import EasyDict as edict
from configs import configs
import torch.nn as nn
from tqdm import tqdm as tqdm
import os

from dataset_with_paths import ImageFolderWithPaths

""" Parse argv --------------------------------------------------------------"""
import argparse

parser = argparse.ArgumentParser(description='Train pytorch classifier.')

parser.add_argument('--config',
                    type=str,
                    default='resnet_0_other_random',
                    help='Name of configuration')

parser.add_argument('--port',
                    type=int,
                    default='6065',
                    help='Port for visdom usage')

parser.add_argument('--num_classes',
                    type=int,
                    default=6,
                    help='TODO')

parser.add_argument('--checkpoint',
                    type=str,
                    default='/raid/user-data/lscheucher/tmp/pytorch_classifier_checkpoints/imageset_5_other_random/resnet_set_0_other_random/18.pth',
                    help='Path to the checkpoint')

parser.add_argument('--resultfile',
                    type=str,
                    default='/raid/user-data/lscheucher/tmp/pytorch_classifier_data/results/res8',
                    help='Path to the checkpoint')


args = parser.parse_args()
opts = configs[args.config]



""" Load model from path """

""" Model, Optimizer, Scheduler"""
""" Model, Optimizer, Scheduler"""
if opts.model == 'SqueezeNet':
    model = models.squeezenet1_0(pretrained=True)
    model.fc = nn.Linear(512, args.num_classes, bias=True)
elif opts.model == 'ResNet18':

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, args.num_classes, bias=True)
elif opts.model == 'ResNet152':

    model = models.resnet18(pretrained=True)
    #Linear(in_features=2048, out_features=args.num_classes, bias=True)
elif opts.model == 'DenseNet169':

    model = models.densenet169(pretrained=True)
    model.classifier = nn.Linear(1664, args.num_classes, bias=True)

elif opts.model == 'DenseNet201':
    model = models.densenet201(pretrained=True)
    model.classifier = nn.Linear(1920, args.num_classes, bias=True)
else:
    raise ValueError("model not supported", opts.model)


model.load_state_dict(torch.load(args.checkpoint))
model.eval()
model = model.to(device)
model.eval()


""" Create validation dataloader """
from torchvision import datasets, models, transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

""" Define appropraite directories for train and validation images"""
if opts.cross_val_phase ==0:

    traindir = os.path.join('/raid/user-data/lscheucher/tmp/pytorch_classifier_data',opts.imageset,'train0')
    valdir = os.path.join('/raid/user-data/lscheucher/tmp/pytorch_classifier_data',opts.imageset,'val0')
else:
    traindir = os.path.join('/raid/user-data/lscheucher/tmp/pytorch_classifier_data',opts.imageset,'train1')
    valdir = os.path.join('/raid/user-data/lscheucher/tmp/pytorch_classifier_data',opts.imageset,'val1')




eval_dataset = ImageFolderWithPaths(
    valdir,
    transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        normalize,
    ]))


eval_sampler = torch.utils.data.SequentialSampler(eval_dataset)

eval_loader = torch.utils.data.DataLoader(
    eval_dataset, batch_size=opts.batchsize, shuffle=False,
    num_workers=20, pin_memory=True, sampler=eval_sampler)

class2idx = eval_dataset.class_to_idx
idx2class = {val:key for key,val in class2idx.items()}

"""
tset = eval_dataset.imgs
len(tset)
Out[11]: 232364


"""


preds = {
    'img_path': [],
    'sub_idx': [],
    'classname': [],
    'pred_classname': [],
    'top': [],
    'left': [],
    'right': [],
    'bottom': []
}

import random
random.seed(0);
idx_list = random.sample(range(len(eval_dataset)), 200)


all_paths = []
all_classnames = []
all_pred_classnames = []

for inputs, labels, paths  in tqdm(eval_loader):

    #print(1)
    inputs, labels= inputs.to(device), labels.to(device)
    #print(2)
    pred = model.forward(inputs)
    #print(3)
    prediction, index = pred.max(1)
    #print(4)
    #store the eval results
    pred_classnames = [idx2class[int(idx)] for idx in index]
    classnames      = [idx2class[int(idx)] for idx in labels]

    all_pred_classnames+=pred_classnames
    all_classnames+=classnames
    all_paths+=paths
    #break

val_annotations = pd.DataFrame(index=all_paths, columns= ['classname', 'pred_classname', 'confidence'], data= [[a,b] for a,b in zip(all_classnames, pred_classnames)])
checkpoint_dir, checkpoint_name = args.checkpoint.rsplit('/', maxsplit=1)
result_file_path = os.path.join(checkpoint_dir,'val_'+checkpoint_name+'.csv')

val_annotations.to_csv(result_file_path)
print("Wrote results to", result_file_path)

print("Done")

"""
imgpath, subidx, classname, predclassname, confidence
...
...
"""
