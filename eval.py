
import torch
from torchvision import datasets, models, transforms
from mydataset import MyDataset
import pandas as pd

import py3nvml
ngpus = py3nvml.grab_gpus(num_gpus=1, gpu_fraction=0.95, gpu_select=range(1,8))
device = 'cuda:0'

pickle_file = '/raid/user-data/lscheucher/projects/bounding_box_classifier/full_object_index.pickle'
from collections import namedtuple
IndexEntry = namedtuple('IndexEntry', ['img_path', 'sub_idx', 'classname', 'left', 'top', 'right', 'bottom'], verbose=False)
from easydict import EasyDict as edict
from configs import configs, index2name, name2index

""" Parse argv --------------------------------------------------------------"""
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

parser.add_argument('--checkpoint',
                    type=str,
                    default='./checkoints/long_train_1.0',
                    help='Path to the checkpoint')

args = parser.parse_args()
opts = edict(configs[args.config])



""" Load model from path """
PATH = './checkpoints/test1.2'
model = models.squeezenet.SqueezeNet(num_classes=5)
model.load_state_dict(torch.load(PATH))
model.eval()
model = model.to(device)


""" Create validation dataloader """
dataset_val   = MyDataset(pickle_file=pickle_file, mode='val')
dataset_loader_val = torch.utils.data.DataLoader(dataset_val,
                                             batch_size=100, shuffle=True,
                                             num_workers=40)



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
idx_list = random.sample(range(len(dataset_val.data)),200)
for idx in idx_list:
#for idx in range(len(dataset_val)):
    entry, img, label = dataset_val.get_item_eval(idx)
    print("Label:", label)

    img = img.to(device)

    pred = model.forward(img[None, :, :, :])

    prediction, index = pred.max(1)
    #tensor([[0.0000, 0.5754, 0.0000, 0.0239, 0.0000]], grad_fn= < ViewBackward >)

    #store the eval results
    pred_classname = index2name[int(index)]
    preds['pred_classname'].append(pred_classname)
    for key, val in entry.items():
        preds[key].append(val)
    print(idx)


from collections import Counter
eval_stat = pd.DataFrame.from_dict(preds)
print(preds)
print("Success")

print("Counter ground truth classes:")
print(Counter(eval_stat.classname))

print("Counter prediction classes:")
print(Counter(eval_stat.pred_classname))



""" Evaluate classifications class-wise """
for classname in name2index.keys():
    num_correct_preds = sum((eval_stat.classname==classname) & (eval_stat.pred_classname == classname))
    num_wrong_preds = sum((eval_stat.classname == classname) & (eval_stat.pred_classname != classname))
    print(classname, num_correct_preds, num_wrong_preds)


"""
test = next(iter(dataset_loader_val))
test[0].shape
Out[6]: torch.Size([100, 3, 224, 224])
test[1].shape
Out[7]: torch.Size([100])
"""