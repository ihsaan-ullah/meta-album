from collections import defaultdict
import numpy as np

import torch
from torchvision import transforms
from avalanche.benchmarks.generators import nc_benchmark
from avalanche.benchmarks.utils import AvalancheDataset, AvalancheSubset

from DataLoader.data_loader_standard import StandardDataset

from typing import Tuple, List, Any

# Define variables
SPLITS = Tuple[AvalancheSubset, AvalancheSubset]
DATASETS = Tuple[List[AvalancheSubset], List[AvalancheSubset]]


class ContinualLearningDataset:
    
    def __init__(self,
                 dataset_list: List[str],
                 img_size: int,
                 train_test_ratio: float = 0.7,
                 random_seed: int = 93) -> None:
        self.dataset_list = dataset_list
        self.img_size = img_size
        self.train_test_ratio = train_test_ratio
        self.random_seed = random_seed
    
    def get_scenario(self, sequence_idx: int) -> Any:
        train_datasets, test_datasets = self.get_datasets(sequence_idx) 

        transform = self.get_transform()

        scenario = nc_benchmark(train_datasets, test_datasets, task_labels=True,
            one_dataset_per_exp=True, shuffle=False, seed=self.random_seed, 
            n_experiences=3, class_ids_from_zero_from_first_exp=True, 
            train_transform=transform["train"], eval_transform=transform["test"]
        )
        
        return scenario
    
    def get_datasets(self, sequence_idx: int) -> DATASETS:
        dataset_sequence = self.get_dataset_sequence(sequence_idx) 

        train_datasets = list()
        test_datasets = list()
        for dataset_path in dataset_sequence:
            dataset = StandardDataset(dataset_path)

            # Sanity check
            assert issubclass(type(dataset), torch.utils.data.Dataset), \
                "Dataset implementation is not correct, it must be a " + \
                "subclass of torch.utils.data.Dataset."
            assert hasattr(dataset, "targets"), "Dataset implementation " +\
                "is not correct, it lacks the 'targets' attribute."

            # Transform the dataset to a dedicated format
            dataset = AvalancheDataset(dataset)
            (train_dataset, 
             test_dataset) = self.train_test_split_by_class(dataset)
            train_datasets.append(train_dataset)
            test_datasets.append(test_dataset)
        
        return train_datasets, test_datasets
    
    def get_dataset_sequence(self, sequence_idx: int) -> List[str]:
        sequences = self.latin_square(len(self.dataset_list))
        permutation = sequences[sequence_idx]
        dataset_sequence = [self.dataset_list[idx] for idx in permutation]
        return dataset_sequence
    
    def latin_square(self, size: int) -> np.array:
        items = list(range(size))
        ls = [items[i:] + items[:i] for i in range(len(items))]
        ls = np.array(ls)
        np.random.shuffle(ls)
        np.random.shuffle(ls.T)    
        return ls
    
    def train_test_split_by_class(self, dataset) -> SPLITS:
        indices = defaultdict(list)
        for idx, data in enumerate(dataset):
            indices[data[1]].append(idx)

        random_gen = np.random.default_rng(self.random_seed)
        for label in indices.keys():
            indices[label] = sorted(indices[label])
            random_gen.shuffle(indices[label])

        train_indices = list()
        test_indices = list()
        for label in indices.keys():
            n = len(indices[label])
            train_n = int(n * self.train_test_ratio)

            train_indices.extend(indices[label][:train_n])
            test_indices.extend(indices[label][train_n:])

        # Shuffle one more time to prevent continual learning tasks from having 
        # unwanted shifting label distribution
        random_gen = np.random.default_rng(self.random_seed//2)
        random_gen.shuffle(train_indices)
        random_gen.shuffle(test_indices)

        train_dataset = AvalancheSubset(dataset, train_indices)
        test_dataset = AvalancheSubset(dataset, test_indices)

        return train_dataset, test_dataset

    def get_transform(self) -> dict:
        transform = {
            "train": transforms.Compose(
                [lambda x: x.convert("RGB"),
                transforms.Resize([self.img_size, self.img_size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])]),
            "test": transforms.Compose(
                [lambda x: x.convert("RGB"),
                transforms.Resize([self.img_size, self.img_size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])
        }
        return transform
