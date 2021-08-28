import io
import os
import h5py
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor

class Data:
    MIN = 1
    CUB = 2

class BatchDataset(Dataset):
    def __init__(self, root_dir, dataset_spec=Data.MIN, transform=None, split="train"):
        self.data_handle = None
        self.data = None
        self.dataset_specifier = dataset_spec
        self.samples = []
        if split == "all":
            self.classes_per_split = []
            for sp in ["train", "val", "test"]:
                self.read_data(root_dir, split=sp)
                if len(self.classes_per_split) == 0:
                    self.classes_per_split.append(len(self.samples))
                else:
                    self.classes_per_split.append( len(self.samples) - sum(self.classes_per_split) )
        else:
            self.read_data(root_dir, split=split)
        self.num_classes = len(self.samples)
        self.num_samples = sum([len(x) for x in self.samples])
        self.transform = transform
        print("Num classes:", self.num_classes)

        # CUB has irregular number of images per class; so make a target map from 
        # example ID -> target class
        if self.dataset_specifier != Data.MIN:
            self.target_map = [0 for _ in range(self.num_samples)]
            # maps target(class) idx -> first index in self.target_map with that class 
            self.offset = [0 for _ in range(self.num_classes)]
            idx = 0
            for cid, class_images in enumerate(self.samples):
                self.offset[cid] = idx
                for j in range(len(class_images)):
                    self.target_map[idx] = cid
                    idx += 1
         
    def read_data(self, root_dir, split="train"):
        data_file = os.path.join(root_dir, f'{split}_data.hdf5')
        self.data_handle = h5py.File(data_file, 'r')
        self.data = self.data_handle['datasets']
        for class_id in self.data.keys():
            samples = np.array(self.data[class_id])
            self.samples.append(samples)
        self.close()
        
    def close(self):
        if not self.data_handle is None:
            self.data_handle.close()
            self.data = None
            self.data_handle = None
            
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.dataset_specifier == Data.MIN:
            target = idx % self.num_classes
            sample_id = (idx - target) // self.num_classes
            image = Image.fromarray(self.samples[target][sample_id])
            target = torch.LongTensor([target])
        else:
            # CUB is stored in different format -.- - this code is required to read the images
            idx = idx % self.num_samples
            target_idx = self.target_map[idx]
            sample_id = idx - self.offset[target_idx]
            image = Image.open(io.BytesIO(self.samples[target_idx][sample_id])).convert('RGB')
            target = torch.LongTensor([target_idx])
        
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def cycle(iterable):
    while True:
        for x in iterable:
            yield x