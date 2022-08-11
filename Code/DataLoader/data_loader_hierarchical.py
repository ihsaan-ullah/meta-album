# --------------------
# Imports

import os
import glob
from torch.utils.data import Dataset,DataLoader, random_split
from torchvision import transforms
from PIL import Image
import pandas as pd
from pandas.api.types import is_numeric_dtype
import json



# --------------------
# Standard Dataset
# Inherited from torch.utils.data.Dataset

class HierarchicalMiniDataset(Dataset):
    """
    PyTorch dataset for super-categories"
    """

    def __init__(self, df, dataset_directory, image_column_name, category_column_name,  transform=None, image_mode="RGB"):
        super().__init__()


        # default : RGB - modifiable for datasets with images other than RGB
        self.image_mode = image_mode

        # default: None - used to transform image 
        self.transform = transform

        self.images = df[image_column_name].to_list()

        # convert label column to strings
        if is_numeric_dtype(df[category_column_name]):
            df[category_column_name] = df[category_column_name].apply(str)

        self.labels = df[category_column_name].to_list()

        self.images_directory = os.path.join(dataset_directory, "images")

        
        # map each raw label to a label (int starting from 0)
        self.numerical_labels = dict()
        for lbl in self.labels:
            if lbl not in self.numerical_labels:
                self.numerical_labels[lbl] = len(self.numerical_labels)
        
        self.categories = self.numerical_labels.keys()


        self.nb_classes = len(self.categories)
        self.data_len = len(self.images)


    def __len__(self):
        return self.data_len

    def __getitem__(self, index):

        img_name = self.images[index]
        img_path = os.path.join(self.images_directory, img_name)

        #load image 
        img = Image.open(img_path).convert(self.image_mode)

        #label 
        label = self.labels[index]
        numerical_label = self.numerical_labels[label]
        
        if self.transform is not None:
            img = self.transform(img)

        return img, numerical_label

   

class HierarchicalDataset():
   
    def __init__(self, dataset_directory, dataset_name, batch_size):

        # Used in preparing data
        self.seed = 33

        self.batch_size = batch_size

        # points to the dataset directory
        self.dataset_directory = dataset_directory
        self.dataset_name = dataset_name

        
        assert os.path.exists(
            dataset_directory), "[-] Dataset path {} not found.".format(dataset_directory)

        # Read Info JSON file
        self.read_info_json()

        # Read CSV file
        self.read_labels_csv()

        assert self.is_valid_for_hierarchical_classification(), "[-] Dataset {} has no super categories.".format(self.dataset_name)
        
        self.construct_dataset()
        


    def construct_dataset(self):
        
        
        # Get super categories
        self.super_categories = sorted(self.df[self.super_category_column_name].unique())

        self.super_categories_datasets = []
        self.super_categories_dataloaders = []
        self.super_categories_datasizes = []
        

        # Loop over all super-categories to create a Pytorch dataset (one df for each super-category)
        for super_category in self.super_categories:
            
            super_category_df = self.df[self.df[self.super_category_column_name] == super_category].groupby(self.category_column_name).sample(n=self.min_images_per_category, random_state=self.seed)
            
            #
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])

            
            

            category_dataset = HierarchicalMiniDataset(df=super_category_df, 
                                dataset_directory=self.dataset_directory, image_column_name=self.image_column_name, 
                                category_column_name=self.category_column_name, transform=transform)

            

            dataset_size = len(category_dataset)
            test_size = int(0.5 * dataset_size)
            train_size = dataset_size - test_size
            train_dataset, test_dataset = random_split(category_dataset,
                                               [train_size, test_size])



            super_category_dataloaders = {
                'test':DataLoader(
                    test_dataset, 
                    batch_size=self.batch_size,
                    shuffle=True,
                ),
                'train':DataLoader(
                    train_dataset, 
                    batch_size=self.batch_size,
                    shuffle=True,
                )
            }
            
            super_category_datasetsizes = {
                'test': len(test_dataset),
                'train':len(train_dataset)
            }

            self.super_categories_dataloaders.append(super_category_dataloaders)
            self.super_categories_datasizes.append(super_category_datasetsizes)
            self.super_categories_datasets.append(category_dataset)
        
        

    def read_info_json(self) -> None:
        info_json_path = os.path.join(
            self.dataset_directory, "info.json")
        with open(info_json_path, "r") as f:
            info_json = json.load(f)

        # "FILE_NAME"
        self.image_column_name = info_json["image_column_name"]
        
        # "CATEGORY"
        self.category_column_name = info_json["category_column_name"]
        
        # "SUPER_CATEGORY"
        self.super_category_column_name = info_json["super_category_column_name"]
        
        # "HAS_SUPER_CATEGORY"
        self.has_super_categories = info_json["has_super_categories"]

        # Min images per cat
        self.min_images_per_category = info_json["minimum_images_per_category"]

    def read_labels_csv(self):
        csv_path = os.path.join(
            self.dataset_directory, "labels.csv")
        self.df = pd.read_csv(csv_path, sep=",", encoding="utf-8")
        

    
    def is_valid_for_hierarchical_classification(self):
        return self.has_super_categories



    def get_super_categories(self):
        return self.super_categories

    def get_categories(self, super_category):
        index = self.super_categories.index(super_category)
        return self.super_categories_datasets[index].categories
    
    def get_super_category_dataloaders(self, super_category):
        index = self.super_categories.index(super_category)
        return self.super_categories_dataloaders[index]

    def get_super_category_dataset(self, super_category):
        index = self.super_categories.index(super_category)
        return self.super_categories_datasets[index]

    def get_super_category_datasizes(self, super_category):
        index = self.super_categories.index(super_category)
        return self.super_categories_datasizes[index]


    def get_super_categories_count(self):
       
        
        str0 = "# -------------------------------- #"
        str1 = "# Total super-categories : {}".format(len(self.super_categories))
        str2 = "# -------------------------------- #"
  
        
        return "{0}\n{1}\n{2}\n".format(str0, str1, str2)

    def get_super_category_statistics(self, super_category):
        index = self.super_categories.index(super_category)
        dataset = self.super_categories_datasets[index]
        
        
        str0 = "# -------------------------------- #"
        str1 = "Super Category : {}".format(super_category)
        str2 = "Total Categories : {}".format(dataset.nb_classes)
        str3 = "Categories:"
        str4 = ""
        for cat in dataset.categories:
            str4 += "\t{}\n".format(cat)
        str5 = "Total data points : {}".format(len(dataset))
        str6 = "Train data points : {}".format(int(len(dataset)/2))
        str7 = "Test data points : {}".format(int(len(dataset)/2))
        str8 = "# -------------------------------- #"
        

        return "{0}\n{1}\n{2}\n{3}\n{4}\n{5}\n{6}\n{7}\n{8}\n".format(
            str0, str1, str2, str3, str4, str5, str6, str7, str8)
        
        




