# code inspired from https://github.com/yinboc/prototypical-network-pytorch
import os.path as osp
import pandas as pd
import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class FewshotDataset(Dataset):

    def __init__(self, img_path, md_path, label_col, file_col, allowed_labels, 
                 img_size=128):
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
        self.img_paths = np.array([osp.join(self.img_path, x) for x in 
            self.md[file_col]])
        # transform string labels into non-negative integer IDs
        self.labels = []
        for label in self.md[label_col]:
            if not label in label_to_id:
                label_to_id[label] = id
                id += 1
            self.labels.append(label_to_id[label])
        self.labels = np.array(self.labels)
        self.transform = transforms.Compose([transforms.Resize((img_size,
            img_size)), transforms.ToTensor()])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, i):
        path, label = self.img_paths[i], self.labels[i]
        image = self.transform(Image.open(path))
        return image, torch.LongTensor([label]).squeeze()


class CategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for _ in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch


def process_labels(batch_size, num_classes):
    return torch.arange(num_classes).repeat(batch_size//num_classes).long()


def extract_info(data_dir, dataset):
    info = {"img_path": f"{data_dir}/{dataset}/images/",
            "metadata_path": f"{data_dir}/{dataset}/labels.csv",
            "label_col": "CATEGORY",
            "file_col": "FILE_NAME"}
    return info["label_col"], info["file_col"],\
           info["img_path"], info["metadata_path"]


def train_test_split_classes(classes, test_split=0.3):
    classes = np.unique(np.array(classes))
    np.random.shuffle(classes)
    cut_off = int((1-test_split)*len(classes))
    return classes[:cut_off], classes[cut_off:]
