import os.path as osp
import pandas as pd
import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# code inspired from https://raw.githubusercontent.com/yinboc/prototypical-network-pytorch/master/mini_imagenet.py
class FewshotDataset(Dataset):

    def __init__(self, img_path, md_path, label_col, file_col, allowed_labels, img_size=128):
        # path to image folder
        self.img_path = img_path
        # path to meta-data file
        self.md_path = md_path
        # column of meta-data file that contains label information
        self.label_col = label_col
        self.file_col = file_col
        # labels to use  
        self.allowed_labels = allowed_labels

        self.md = pd.read_csv(self.md_path)
        # select only data with permissible labels
        self.md = self.md[self.md[label_col].isin(self.allowed_labels)]

        label_to_id = dict(); id = 0
        self.img_paths = np.array([osp.join(self.img_path, x) for x in self.md[file_col]])
        # transform string labels into non-negative integer IDs
        self.labels = []
        for label in self.md[label_col]:
            if not label in label_to_id:
                label_to_id[label] = id
                id += 1
            self.labels.append(label_to_id[label])
        self.labels = np.array(self.labels)
        self.transform = transforms.Compose([transforms.Resize((img_size,img_size)), transforms.ToTensor()])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, i):
        path, label = self.img_paths[i], self.labels[i]
        image = self.transform(Image.open(path))
        return image, torch.LongTensor([label]).squeeze()

