import os
import glob
import json
from PIL import Image
import pandas as pd

import torch
from torch.utils.data import Dataset

from typing import Tuple, List, Any

# Data types definitions
DATA_FRAME = pd.core.frame.DataFrame

class StandardDataset(Dataset):
    """
    PyTorch dataset with the field "targets"
    """

    def __init__(self, 
                 dataset_directory: str, 
                 transform: Any = None, 
                 image_mode: str = "RGB") -> None:
        super().__init__()

        self.dataset_directory = dataset_directory
        self.image_mode = image_mode
        self.transform = transform

        assert os.path.exists(
            dataset_directory), f"Dataset path {dataset_directory} not found."

        self.items = self.construct_items()
        self.add_field_targets()

        self.nb_classes = int(max(self.targets)) + 1

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        img_path, _, label = self.items[idx]
        img = Image.open(img_path).convert(self.image_mode)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def construct_items(self) -> List[list]:
        # Get raw labels
        self.read_info_json()
        df = self.read_labels_csv()

        search_pattern = os.path.join(self.dataset_directory, "images", "*.*")
        items = list()

        # Get list of lists, each list contains the image path and the label
        for path in sorted(glob.glob(search_pattern)):
            base_path = os.path.basename(path)
            file_ext = os.path.splitext(base_path)[-1].lower()
            if file_ext in [".png", ".jpg", ".jpeg"]:
                info = [path, df.loc[base_path, self.category_column_name]]
                items.append(info)

        # Map each raw label to a label (int starting from 0)
        self.raw_label2label = dict()
        for item in items:
            if item[1] not in self.raw_label2label:
                self.raw_label2label[item[1]] = len(self.raw_label2label)
            item.append(self.raw_label2label[item[1]])
        return items

    def read_info_json(self) -> None:
        info_json_path = os.path.join(self.dataset_directory, "info.json")
        with open(info_json_path, "r") as f:
            info_json = json.load(f)
        # "FILE_NAME"
        self.image_column_name = info_json["image_column_name"]
        # "CATEGORY"
        self.category_column_name = info_json["category_column_name"]

    def read_labels_csv(self) -> DATA_FRAME:
        csv_path = os.path.join(self.dataset_directory, "labels.csv")
        df = pd.read_csv(csv_path, sep=",", encoding="utf-8")
        df = df.loc[:, [self.image_column_name, self.category_column_name]]
        df.set_index(self.image_column_name, inplace=True)
        return df

    def add_field_targets(self) -> None:
        """
        The targets field is available in nearly all torchvision datasets. It 
        must be a list containing the label for each data point (usually the y 
        value).
        
        https://avalanche.continualai.org/how-tos/avalanchedataset/creating-avalanchedatasets
        """
        self.targets = [item[2] for item in self.items]
        self.targets = torch.tensor(self.targets, dtype=torch.int64)

    def get_raw_label2label_dict(self) -> dict:
        return self.raw_label2label
