#!/usr/bin/env python
# coding: utf-8


#=================================================
# Imports
#=================================================
import sys
import argparse

import os
import json

import random
import math
import time
import datetime
import copy


import pandas as pd
import numpy as np

import cv2

import gc



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader







#=================================================
# Arguments
#=================================================

parser=argparse.ArgumentParser()

parser.add_argument('--DATASET_PATH', required=True, 
    help='Path of the dateset (images,csv,info)')


parser.add_argument('--TABULAR_PATH', type=str, required=True, 
    help="Path of the directory where the tabular data will bee saved")



parser.add_argument('--USE_NORMALIZATION', action='store_true', default=False, 
    help='Normalize the input images according to the way neural networks were pretrained on ImageNet')


parser.add_argument('--RETRAIN', action='store_true', default=False, 
    help='Flag to retrain the network on your data before generating tabular data')

parser.add_argument('--RETRAIN_DATASET_PATH', default=None, 
    help='The path to directory where the training images are stored (128x128x3)')


args=parser.parse_args()





#=================================================
# Define Errors
#=================================================
class TabularErrors:
    def FileError(self,file):
        raise IOError('[-] File Not Found : '+file)
        exit()
    def DirectoryError(self,directory):
        raise IOError('[-] Directory Not Found : '+directory)
        exit()
    def ColumnError(self,column):
        raise ValueError('[-] Column Not Found : '+column)
        exit()
err = TabularErrors()




#=================================================
# Settings from Argumeents
#=================================================

# Path of the dataset which contains images, labels.csv and info.json
DATASET_PATH = args.DATASET_PATH


# Path of the directory where the tabular data will bee saved
TABULAR_PATH = args.TABULAR_PATH




# Normalize the input images according to the way neural networks were pretrained on ImageNet
USE_NORMALIZATION = args.USE_NORMALIZATION

# To retrain the network on your data before generating tabular data
RETRAIN = args.RETRAIN

# The path to directory where the training images are stored (128x128x3)
RETRAIN_DATASET_PATH = args.RETRAIN_DATASET_PATH

# seed for generating super-categories by the same random combination of categories
SEED = 42






# Path of the json.info file
JSON_PATH = os.path.join(DATASET_PATH, "info.json")

# Path of the CSV file which contains the label and image name
CSV_PATH = os.path.join(DATASET_PATH, "labels.csv")



#=================================================
# Create Tabular
#=================================================

if not os.path.exists(TABULAR_PATH):
    os.makedirs(TABULAR_PATH)
    
    
    

#=================================================
# Check Directories and Files
#=================================================

# Dataset Directory
if not os.path.exists(DATASET_PATH):
    err.DirectoryError(DATASET_PATH)
    

#Check JSON file
if not os.path.isfile(JSON_PATH):
    err.FileError(JSON_PATH)

# Check CSV File
if not os.path.isfile(CSV_PATH):
    err.FileError(CSV_PATH)



# Check Tabular Directory
if not os.path.exists(TABULAR_PATH):
    err.DirectoryError(TABULAR_PATH)

    
    
    
#=================================================
# Check Retrain Directories and Files
#=================================================
    
if RETRAIN:
    # Path of the json.info file
    RETRAIN_JSON_PATH = os.path.join(RETRAIN_DATASET_PATH, "info.json")

    # Path of the CSV file which contains the label and image name
    RETRAIN_CSV_PATH = os.path.join(RETRAIN_DATASET_PATH, "labels.csv")


    # Dataset Directory
    if not os.path.exists(RETRAIN_DATASET_PATH):
        err.DirectoryError(RETRAIN_DATASET_PATH)
        
    #Check JSON file
    if not os.path.isfile(RETRAIN_JSON_PATH):
        err.FileError(RETRAIN_JSON_PATH)

    # Check CSV File
    if not os.path.isfile(RETRAIN_CSV_PATH):
        err.FileError(RETRAIN_CSV_PATH)
        
        
        

#=================================================
# Read JSON
#=================================================

f = open (JSON_PATH, "r")
info = json.loads(f.read())


#=================================================
# Read Retrain JSON
#=================================================

retrain_info = None
if RETRAIN:
    f = open (RETRAIN_JSON_PATH, "r")
    retrain_info = json.loads(f.read())
    
    
    
#=================================================
# Settings from info JSON file
#=================================================


# True if CSV is tab separated otherwise false
CSV_WITH_TAB = info["csv_with_tab"]


# Path of the directory where images to be used in this experiement are saved
if info["images_in_sub_folder"]:
    IMAGE_PATH = os.path.join(DATASET_PATH, "images")
else:  
    IMAGE_PATH = DATASET_PATH

    
# category column name in csv
CATEGORY_COLUMN = info["category_column_name"]

# image column name in csv
IMAGE_COLUMN = info["image_column_name"]



    
#=================================================
# Settings from Retrain info JSON file
#=================================================
if RETRAIN:
    RETRAIN_CSV_WITH_TAB = retrain_info["csv_with_tab"]
    
    if retrain_info["images_in_sub_folder"]:
        RETRAIN_IMAGE_PATH = os.path.join(RETRAIN_DATASET_PATH, "images")
    else:  
        RETRAIN_IMAGE_PATH = RETRAIN_DATASET_PATH
        
        
    # category column name in csv
    RETRAIN_CATEGORY_COLUMN = retrain_info["category_column_name"]

    # image column name in csv
    RETRAIN_IMAGE_COLUMN = retrain_info["image_column_name"]
    
    
    
    
#=================================================
# Load CSV
#=================================================

if CSV_WITH_TAB:
    data = pd.read_csv(CSV_PATH, sep="\t", encoding="utf-8") 
else:
    data = pd.read_csv(CSV_PATH)

print("Data Shape : ", data.shape)




#=================================================
# Load Retrain CSV
#=================================================

if RETRAIN:
    
    if RETRAIN_CSV_WITH_TAB:
        retrain_data = pd.read_csv(RETRAIN_CSV_PATH, sep="\t", encoding="utf-8") 
    else:
        retrain_data = pd.read_csv(RETRAIN_CSV_PATH)
    
    print("Data Shape : ", retrain_data.shape)
    

#=================================================
# Prepare Data
#=================================================
def prepare_data(csv_data, cat_col, img_col):
    dictt = {}
  
    
    csv_data['label_cat'] = csv_data[cat_col].astype('category')

    dictt['categories'] = csv_data['label_cat'].cat.categories.values
    dictt['images'] = csv_data[cat_col].value_counts().values

    dictt['labels'] = csv_data[cat_col].values
    dictt['labels_num'] =  csv_data['label_cat'].cat.codes.values
    dictt['data'] = csv_data[img_col].values

    
    return dictt


prepared_data = prepare_data(data, CATEGORY_COLUMN, IMAGE_COLUMN)

retrain_prepared_data = None
if RETRAIN:
    retrain_prepared_data = prepare_data(retrain_data, RETRAIN_CATEGORY_COLUMN, RETRAIN_IMAGE_COLUMN)

    
    
    
    
    
    
    
    
    
    
    
#=================================================
# Device
#=================================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")





#=================================================
# Data
#=================================================
class ImgDataset(Dataset):
    def __init__(self, dataset_images, dataset_labels, transform):

        # Transforms
        self.transform = transform
        
        self.images = dataset_images
        self.labels = dataset_labels
        
        self.data_len = len(self.images)

    def __getitem__(self, index):
        
        
        single_img = self.images[index]
        img_transformed = torch.from_numpy(single_img).long()
        img_transformed = img_transformed.permute(2, 0, 1)
        img_transformed = torch.from_numpy(np.array(img_transformed)).float() / 255.

        single_label = self.labels[index]
        single_label = single_label.astype(np.compat.long)
        
        return img_transformed, single_label

    def __len__(self):
        return self.data_len
    
    
    
    
    
#=================================================
# Make Dataset
#=================================================
def make_dataset(data_set, batch_size=64, shuffle=False):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    if not USE_NORMALIZATION:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    
    
    train_ds=ImgDataset(data_set['images'],data_set['labels_num'], transform)
    
    
    
  
    print("Data: ", len(train_ds))
   
    
    data_stats = {
        "train_images" : len(train_ds)
    }
    
    
    dataloaders = {
        'train':DataLoader(
            train_ds, 
            batch_size=batch_size,
            shuffle=shuffle,
        )
    }

    dataset_sizes = {
        'train':len(train_ds)
    }
    
    return dataloaders, dataset_sizes, data_stats



#=================================================
# Model
#=================================================

def getModel(output_features=32,only_train_last_layer=True):
    model = models.resnet18(pretrained=True)

    if only_train_last_layer:
        for param in model.parameters():
            param.requires_grad = False
        
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, output_features)
    return model.to(device)



#=================================================
# Training Loop
#=================================================
def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=5):
    
    
    print("--------------------------------------------")
    print("Training")
    print("--------------------------------------------")
    
    since = time.time()


    print("Epoch: ", end=" ")
    for epoch in range(num_epochs):
        print(epoch, end=" ")
        

        # Each epoch has a training phase

        model.train()


        # Iterate over data.
        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):

                outputs = model(inputs)
#                 probabilities = F.softmax(outputs, dim=1)
#                 _, preds = torch.max(outputs, 1)

                loss = criterion(outputs, labels)


                # backward + optimize
                
                loss.backward()
                optimizer.step()


           

        # end dataloader loop

        scheduler.step()

    time_elapsed = time.time() - since
    print()
    print()
    
    training_time = '{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)
    
    print('Training complete in: {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    print()



    return model


#=================================================
# Get Predictions
#=================================================
def get_predictions(model, dataloaders, dataset_sizes):
    
    
    print("--------------------------------------------")
    print("Extracting Predictions")
    print("--------------------------------------------")
    
    model.eval()

    all_outputs = []
    # Iterate over data.
    for inputs, labels in dataloaders['train']:
        inputs = inputs.to(device)
        outputs = model(inputs).detach().numpy()
        all_outputs.append(outputs)

    return np.vstack(all_outputs)       



#=================================================
# Magic Happens here
#=================================================
def load_images(data_set, images_path):
        
    images = []
    
    
    for image_name in data_set['data']:
        file = images_path+"/"+image_name
        img = cv2.imread(file)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img_rgb)

   
    data_set['images'] = images
   
    return data_set



def train_network(data_set, images_path):
    
    data_set = load_images(data_set,images_path)
    

    dataloaders, dataset_sizes, data_stats = make_dataset(data_set,shuffle=True)
    model = getModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    trained_model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, exp_lr_scheduler, num_epochs=10)
    
  
    
    #CleanUp
    del dataloaders
    del model
    del criterion
    del optimizer
    del exp_lr_scheduler
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return trained_model





#=================================================
# Training Starts here
#=================================================
trained_model = None
if RETRAIN:
    trained_model = train_network(retrain_prepared_data, RETRAIN_IMAGE_PATH)
else:
    trained_model = getModel()



#=================================================
# Extract and save tabular data
#=================================================
def extract_and_save_tabular_data(trained_model=trained_model):
    
    data_set = load_images(prepared_data,IMAGE_PATH)
    dataloaders, dataset_sizes, data_stats = make_dataset(data_set,batch_size=1)
    tabular_data = get_predictions(trained_model, dataloaders, dataset_sizes)
    
    #columns 
    columns = ['feat_'+str(i+1) for i in range(tabular_data.shape[1])]
    
    #generating dataframe
    tabular_df = pd.DataFrame(data=tabular_data, columns=columns)
    
    #adding labels
    tabular_df['CATEGORY'] = data[CATEGORY_COLUMN]
    
    #save to tabular directory
    tabular_csv_path = os.path.join(TABULAR_PATH,'labels.csv')
    tabular_df.to_csv(tabular_csv_path, index=False)
    
    print("--------------------------------------------")
    print("Tabular Dataset Saved")
    print("--------------------------------------------")


extract_and_save_tabular_data()

#Finish