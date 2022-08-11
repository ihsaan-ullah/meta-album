import os.path
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.utils import check_random_state
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class MetaAlbumDataset(Dataset):

    def __init__(self, datasets, data_dir, img_size=128):
        if len(datasets) == 1: 
            self.name = datasets[0]
        else:
            self.name = f"Multiple datasets: {','.join(datasets)}"
        self.data_dir = data_dir
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)), transforms.ToTensor()])
            
        self.img_paths = list()
        self.labels = list()
        id_ = 0
        for dataset in datasets:
            (label_col, file_col, 
            img_path, md_path) = self.extract_info(dataset)
            
            metadata = pd.read_csv(md_path)
            
            self.img_paths.extend([os.path.join(img_path, x) for x in 
                metadata[file_col]])
            
            # Transform string labels into non-negative integer IDs
            label_to_id = dict()
            for label in metadata[label_col]:
                if label not in label_to_id:
                    label_to_id[label] = id_
                    id_ += 1
                self.labels.append(label_to_id[label])
        self.labels = np.array(self.labels)
        
        self.idx_per_label = []
        for i in range(max(self.labels) + 1):
            idx = np.argwhere(self.labels == i).reshape(-1)
            self.idx_per_label.append(idx)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, i):
        path, label = self.img_paths[i], self.labels[i]
        image = self.transform(Image.open(path))
        return image, torch.LongTensor([label]).squeeze()
    
    def extract_info(self, dataset):
        info = {"img_path": f"{self.data_dir}/{dataset}/images/",
            "metadata_path": f"{self.data_dir}/{dataset}/labels.csv",
            "label_col": "CATEGORY",
            "file_col": "FILE_NAME"}
        return info["label_col"], info["file_col"], info["img_path"], \
            info["metadata_path"]
    
    
def process_labels(batch_size, num_classes):
    return torch.arange(num_classes).repeat(batch_size//num_classes).long()


def create_datasets(datasets, data_dir):
    torch_datasets = []
    for dataset in datasets:
        torch_dataset = MetaAlbumDataset([dataset], data_dir)
        torch_datasets.append(torch_dataset)
    return torch_datasets


class Task:
    
    def __init__(self, 
                 n_way, 
                 k_shot,
                 query_size, 
                 data, 
                 labels, 
                 dataset):
        self.n_way = n_way
        self.k_shot = k_shot
        self.query_size = query_size
        self.data = data
        self.labels = labels
        self.dataset = dataset
        
        
class DataLoaderCrossDomain:
    def __init__(self, datasets, num_tasks, episodes_config, 
                 test_loader=False):
        self.datasets = datasets
        self.num_tasks = num_tasks
        self.test_loader = test_loader
        self.n_way = episodes_config["n_way"]
        self.min_ways = episodes_config["min_ways"]
        self.max_ways = episodes_config["max_ways"]
        self.k_shot = episodes_config["k_shot"]
        self.min_shots = episodes_config["min_shots"]
        self.max_shots = episodes_config["max_shots"]
        self.query_size = episodes_config["query_size"]
        
    def generator(self, seed):
        if not self.test_loader:
            while True:
                random_gen = check_random_state(seed)
                for _ in range(self.num_tasks):
                    # Select task configuration
                    dataset_idx = random_gen.randint(0, len(self.datasets))
                    dataset = self.datasets[dataset_idx]
                    idx_per_label = dataset.idx_per_label
                    num_classes = len(idx_per_label)
                    n_way, k_shot = self.prepare_task_config(num_classes,
                        random_gen)
                    total_examples = k_shot + self.query_size
                    
                    # Select examples for the task
                    batch = list()
                    classes = random_gen.permutation(num_classes)[:n_way]
                    for c in classes:
                        idx = idx_per_label[c]
                        selected_idx = random_gen.choice(idx, total_examples, 
                            replace=False)
                        batch.append(selected_idx)
                    batch = np.stack(batch).T.reshape(-1)
                    
                    # Load the examples
                    data = list()
                    labels = list()
                    for i in batch:
                        img, label = dataset[i]
                        data.append(img)
                        labels.append(label)
                    data = torch.stack(data)
                    labels = torch.stack(labels)
                    
                    # Return the task
                    task = Task(n_way, k_shot, self.query_size, data, labels, 
                        dataset.name)
                    yield task        
        else:
            for dataset_idx in range(len(self.datasets)):
                random_gen = check_random_state(seed)
                
                # Dataset information
                dataset = self.datasets[dataset_idx]
                idx_per_label = dataset.idx_per_label
                num_classes = len(idx_per_label)
                for _ in range(self.num_tasks):
                    # Select task configuration
                    n_way, k_shot = self.prepare_task_config(num_classes,
                        random_gen)
                    total_examples = k_shot + self.query_size
                    
                    # Select examples for the task
                    batch = list()
                    classes = random_gen.permutation(num_classes)[:n_way]
                    for c in classes:
                        idx = idx_per_label[c]
                        selected_idx = random_gen.choice(idx, total_examples, 
                            replace=False)
                        batch.append(selected_idx)
                    batch = np.stack(batch).T.reshape(-1)
                    
                    # Load the examples
                    data = list()
                    labels = list()
                    for i in batch:
                        img, label = dataset[i]
                        data.append(img)
                        labels.append(label)
                    data = torch.stack(data)
                    labels = torch.stack(labels)
                    
                    # Return the task
                    task = Task(n_way, k_shot, self.query_size, data, labels, 
                        dataset.name)
                    yield task        
                
    def prepare_task_config(self, num_classes, random_gen):
        n_way = self.n_way
        if n_way is None:
            max_ways = num_classes
            if self.max_ways < max_ways:
                max_ways = self.max_ways
            
            n_way = random_gen.randint(self.min_ways, max_ways + 1)
        
        k_shot = self.k_shot
        if k_shot is None:
            k_shot = random_gen.randint(self.min_shots, self.max_shots+1)
        
        return n_way, k_shot
