from torchvision import models
import pandas as pd

import py3nvml
ngpus = py3nvml.grab_gpus(num_gpus=1, gpu_fraction=0.95, gpu_select=range(0,8))
device = 'cuda:0'

pickle_file = '/raid/user-data/lscheucher/projects/bounding_box_classifier/full_object_index.pickle'
from collections import namedtuple
IndexEntry = namedtuple('IndexEntry', ['img_path', 'sub_idx', 'classname', 'left', 'top', 'right', 'bottom'], verbose=False)
from configs import configs
import os

from eval_checkpoint import eval_checkpoint
from plots import bar_chart

""" Parse argv --------------------------------------------------------------"""
import argparse

parser = argparse.ArgumentParser(description='Train pytorch classifier.')

parser.add_argument('--config',
                    type=str,
                    default='densenet169_0_other',
                    help='Name of configuration')

parser.add_argument('--port',
                    type=int,
                    default='6065',
                    help='Port for visdom usage')

parser.add_argument('--checkpoint_0',
                    type=str,
                    default='/raid/user-data/lscheucher/tmp/pytorch_classifier_checkpoints/imageset_5_other/densenet169_set_0_withother/36.pth',
                    help='Path to the checkpoint')
parser.add_argument('--checkpoint_1',
                    type=str,
                    default='/raid/user-data/lscheucher/tmp/pytorch_classifier_checkpoints/imageset_5_other/densenet169_set_1_withother/36.pth',
                    help='Path to the checkpoint')

args = parser.parse_args()
opts = configs[args.config]


""" Create val annotations """
val_annotations_0 = eval_checkpoint(opts, 0, models, args.checkpoint_0)
val_annotations_1 = eval_checkpoint(opts, 1, models, args.checkpoint_1)
val_annotations = pd.concat([val_annotations_0, val_annotations_1])


""" Write results to disk """
outfile = os.path.join(args.checkpoint_0.rsplit('/', maxsplit=1)[0], 'eval_'+args.checkpoint_0.rsplit('/', maxsplit=1)[1].replace('.pth','.csv'))
print("Writing results to:",outfile)
val_annotations.to_csv(outfile)


""" Create corresponding plots """
plotfile = os.path.join(args.checkpoint_0.rsplit('/', maxsplit=1)[0], 'eval_'+args.checkpoint_0.rsplit('/', maxsplit=1)[1].replace('.pth','.html'))
print("Writing plots to:",plotfile)
bar_chart(val_annotations, plotfile)
