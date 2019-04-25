import torch

import pandas as pd
import numpy as np
import pickle
from collections import namedtuple
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
import PIL


class MyDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, pickle_file, mode='train'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        IndexEntry = namedtuple('IndexEntry', ['img_path', 'sub_idx', 'classname', 'left', 'top', 'right', 'bottom'],
                                verbose=False)
        with open(pickle_file, 'rb') as f:
            self.idxs = pickle.load(f)

        self.name2index = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2, 'TrafficSign': 3, 'TrafficSignal': 4}
        self.index2name = { val:key for key,val in self.name2index.items()}

        data = {'img_path': [entry.img_path for entry in self.idxs],
                'sub_idx': [entry.sub_idx for entry in self.idxs],
                'classname': [entry.classname for entry in self.idxs],
                'left': [entry.left for entry in self.idxs],
                'top': [entry.top for entry in self.idxs],
                'right': [entry.right for entry in self.idxs],
                'bottom': [entry.bottom for entry in self.idxs],
                }
        self.data = pd.DataFrame.from_dict(data)
        index = (self.data.classname == 'Car') | \
                (self.data.classname == 'Pedestrian') | \
                (self.data.classname == 'Cyclist') | \
                (self.data.classname == 'TrafficSign') | \
                (self.data.classname == 'TrafficSignal')
        self.data = self.data[index]


        self.balance_data()
        del self.idxs
        del data

        dividor = len(self.data) >> 2
        if mode == 'train':
            ###self.data = self.data.iloc[400:1000]
            self.data = self.data.iloc[dividor:]
        else:
            self.data = self.data.iloc[:dividor]
            ###self.data = self.data.iloc[:400]

        self.transformer = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    # def reshuffle(self):
    #     return 0

    def balance_data(self):
        def classname2num(classname):
            return sum(self.data.classname == classname)

        classnames = set(self.data.classname)
        print("Before balancing")
        for classname in classnames:
            print(classname, sum(self.data.classname == classname) )

        MIN = min([sum(self.data.classname == classname) for classname in classnames])
        MAX = max([sum(self.data.classname == classname) for classname in classnames])

        test = np.where(self.data.classname == classname)[0]
        test = np.random.choice(test, 2)
        subset = self.data.iloc[test]

        subsets = [self.data.iloc[np.random.choice(np.where(self.data.classname == classname)[0], min(2*MIN, classname2num(classname)) )] for classname in classnames]
        self.data = pd.concat(subsets)

        print("After balancing")
        for classname in classnames:
            print(classname, sum(self.data.classname == classname) )

    def __getitem__(self, idx):
        entry = self.data.iloc[idx]
        img = PIL.Image.open(entry.img_path)
        # left, upper, right, and lower
        img = img.crop(box=[entry.left, entry.top, entry.right, entry.bottom])
        img = self.transformer(img)
        # label = torch.eye(4)[self.name2index[entry.classname]]
        # label = torch.empty(3, dtype=torch.long)
        label = torch.tensor(self.name2index[entry.classname], dtype=torch.long)

        return (img, label)

    def get_item_eval(self, idx):
        entry = self.data.iloc[idx]
        img = PIL.Image.open(entry.img_path)
        # left, upper, right, and lower
        img = img.crop(box=[entry.left, entry.top, entry.right, entry.bottom])
        img = self.transformer(img)
        # label = torch.eye(4)[self.name2index[entry.classname]]
        # label = torch.empty(3, dtype=torch.long)
        label = torch.tensor(self.name2index[entry.classname], dtype=torch.long)

        return (entry, img, label)