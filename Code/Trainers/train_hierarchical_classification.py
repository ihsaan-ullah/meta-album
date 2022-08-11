#-----------------------
# Imports
#-----------------------

import sys
import os

# in order to import modules from packages in the parent directory
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import datetime
import warnings
import numpy as np

import random
import time
import argparse

import pandas as pd
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from Models.resnet18 import ResNet18
from DataLoader.data_loader_hierarchical import HierarchicalDataset

#-----------------------
# Arguments
#-----------------------
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Hierachical classification experiment on Meta-Album")
    parser.add_argument("--dataset", type=str, 
                        help="""Dataset ID from Meta-Album or Dataset Directory name. Keep Meta-Album ID in directory name """)
    parser.add_argument("--random_seed", type=int, default=None)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001,
                        help="""learning rate of SGD optimizer.""")
    parser.add_argument("--step_size", type=int, default=7,
                        help="""step size.""")
    parser.add_argument("--gamma", type=float, default=0.1,
                        help="""Gamma""")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="""momentum of SGD optimizer.""")
   
    args = parser.parse_args()
    
        
    return args



#-----------------------
# Hierarchical Classification Training
#-----------------------

class Hyper_Parameters :
    def __init__(self, args):

        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.criterion = nn.CrossEntropyLoss()
        self.lr = args.lr
        self.step_size = args.step_size
        self.gamma = args.gamma
        self.momentum = args.momentum
        self.factor = 0.1
        self.patience = 10

        

class Hierarchical_Classification_Training:

    def __init__(self, args):


        self.args = args

        # Set hyper-parameters
        self.hp = Hyper_Parameters(self.args)

        # Dataset Name
        self.dataset_name = self.args.dataset
        
        # Set data
        self.set_data()

        # Results
        self.primary_results_dir = "Results"
        self.secondary_results_dir = "hierarchical_classification"
        self.run_id = "run_{}_{}".format(str(self.args.random_seed), str(datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")))
        self.res_dir_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                self.primary_results_dir, self.secondary_results_dir, self.dataset_name, self.run_id)
        

        # Set results directory
        self.set_results_dir()
        


        # Set Omniprint Dataset IDs
        self.set_omniPrint_ids()
        

        self.lprint("[!] torch.__version__: {}".format(torch.__version__))

    def lprint(self, text):

        print(text)
        training_logs_path = os.path.join(self.res_dir_path, 'training_logs.txt')
        with open(training_logs_path, 'a') as f:
            f.write(text + "\n")
    
    def set_omniPrint_ids(self):
        self.omniprint_ids = [
            "MD_6",
            "MD_5_BIS",
            "MD_MIX"
        ]
    
    def is_omniprint_dataset(self):

        is_omni = False
        for omni in self.omniprint_ids:
            if omni in self.dataset_name:
                is_omni = True

        return is_omni

    def set_random_seeds(self):
        random_seed = self.args.random_seed
        if random_seed is not None:
            torch.backends.cudnn.deterministic = False
            random.seed(random_seed)
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(random_seed)

    def set_device(self):
        if torch.cuda.is_available():
            self.lprint("[!] Using GPU for PyTorch: {}".format(
                torch.cuda.get_device_name(torch.cuda.current_device())))
        else:
            device = torch.device("cpu")
            self.lprint("[!] Using CPU for PyTorch")
        self.device = device

    def get_model(self, dataset):
        
        resNet18 = ResNet18(number_of_classes= dataset.nb_classes, device=self.device)
        return resNet18.get_model()

    

    def set_data(self):
        
        self.dataset_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Data", self.dataset_name)
        
        self.data = HierarchicalDataset(self.dataset_dir, self.dataset_name, self.hp.batch_size)
        

         


    def set_results_dir(self):
        
        if not os.path.exists(self.res_dir_path):
            os.makedirs(self.res_dir_path)
            self.lprint("[+] Results Directory created : {}".format(self.res_dir_path))
        else:
            self.lprint("[!] Results Directory already exists : {}".format(self.res_dir_path))

    
    def save_experimental_settings(self):

        text_to_save = [
            "Description : This experiment is named 'Hierarchical classification'. It is used for datasets with hierarchical classes i.e. super-categories and categories. Each super-category is used to make a classification task with only the classes which belongs to that super-category. The number of classification tasks are equal to the number of super-categories in the dataset.",
            "Model : ResNet-18",
            "Loss function : {}".format(self.hp.criterion),
            "Learning rate : {}".format(self.hp.lr),
            "Number of epochs : {}".format(self.hp.epochs),
            "Batch size : {}".format(self.hp.batch_size),
        ]

        if self.is_omniprint_dataset():
            text_to_save.extend([
                "Optimizer : Adam",
                "Scheduler : ReduceLROnPlateau",
                "Factor : {}".format(self.hp.factor),
                "Patience : {}".format(self.hp.patience),
            ])
        else:
            text_to_save.extend([
                "Optimizer : SGD",
                "Scheduler : StepLR",
                "Step size : {}".format(self.hp.step_size),
                "Momentum : {}".format(self.hp.momentum),
                "Gamma : {}".format(self.hp.gamma)
            ])
        

        experimental_settings_path = os.path.join(self.res_dir_path, 'experimental_settings.txt')
        with open(experimental_settings_path, 'w') as f:
            f.writelines('\n'.join(text_to_save))
    
    def save_super_categories(self):
        super_categories_path = os.path.join(self.res_dir_path, 'super_categories.txt')
        with open(super_categories_path, 'w') as f:
            f.writelines('\n'.join(self.data.get_super_categories()))

    def save_categories_result(self, super_category, training_time,
                        best_train_acc, best_test_acc, 
                        train_score, train_loss, test_score, test_loss,
                        train_predictions, train_labels, train_predicted_probabilities,
                        test_predictions, test_labels, test_predicted_probabilities):
        
        # Create a super-category directory
        super_category_dir = os.path.join(self.res_dir_path, super_category)
        os.mkdir(super_category_dir)

        # Save categories
        category_path = os.path.join(super_category_dir, "categories.txt")
        with open(category_path, 'w') as f:
            f.writelines('\n'.join(self.data.get_categories(super_category)))

        # Save logs
        logs_path = os.path.join(super_category_dir, "logs.txt")
        with open(logs_path, 'w') as f:
            f.write("Training time : {}\n".format(training_time))
            f.write("Best train accuracy : {}\n".format(best_train_acc))
            f.write("Best test accuracy : {}".format(best_test_acc))


        # Save TRAIN acc and loss
        train_acc_loss_path = os.path.join(super_category_dir, "train_acc_loss.csv")
        with open(train_acc_loss_path, 'w', encoding='UTF8') as f:
            writer = csv.writer(f)

            # write the header
            writer.writerow(['Accuracy', 'Loss'])

            for index, item in enumerate(train_score):
                # write the data
                writer.writerow([train_score[index], train_loss[index]])



        # Save TEST acc and loss
        test_acc_loss_path = os.path.join(super_category_dir, "test_acc_loss.csv")
        with open(test_acc_loss_path, 'w', encoding='UTF8') as f:
            writer = csv.writer(f)

            # write the header
            writer.writerow(['Accuracy','Loss'])

            for index, item in enumerate(train_score):
                
                # write the data
                writer.writerow([test_score[index], test_loss[index]])
        

        # Save TRAIN predictions labels and probabilities
        train_pred_labels_probs_path = os.path.join(super_category_dir, "train_pred_labels_probs.csv")
        with open(train_pred_labels_probs_path, 'w', encoding='UTF8') as f:
            writer = csv.writer(f)

            # write the header
            writer.writerow(['Label', 'Prediction', 'Probabilities'])

            for index, item in enumerate(train_labels):
                
                #prepare data
                data = [train_labels[index], train_predictions[index], train_predicted_probabilities[index]]
                
                # write the data
                writer.writerow(data)


        # Save TEST predictions labels and probabilities
        test_pred_labels_probs_path = os.path.join(super_category_dir, "test_pred_labels_probs.csv")
        with open(test_pred_labels_probs_path, 'w', encoding='UTF8') as f:
            writer = csv.writer(f)

            # write the header
            writer.writerow(['Label', 'Prediction', 'Probabilities'])

            for index, item in enumerate(train_labels):
                
                #prepare data
                data = [test_labels[index], test_predictions[index], test_predicted_probabilities[index]]
                
                # write the data
                writer.writerow(data)

        
    def start_training(self):
        """
        This function is responsible for training the model on each super-category
        """


        self.lprint("\n\n### ------------------------------------------ ###")
        self.lprint("### Training Dataset : {} ".format(self.dataset_name))
        self.lprint("### Random Seed : {} ".format(self.args.random_seed))
        self.lprint("### ------------------------------------------ ###\n")

        

        # Save Experimental settings
        self.save_experimental_settings()

        # Save all super-categories
        self.save_super_categories()

        self.lprint(self.data.get_super_categories_count())
        

        # Loop over all super-categories and perform hierarchical classiciation
        # Classification inside a super-category
        for super_category in self.data.get_super_categories():
            
            # Time
            since = time.time()

            self.lprint(self.data.get_super_category_statistics(super_category))
            
          

            # Phases
            phases = ['train', 'test']

            # Data loaders
            super_category_dataloaders = self.data.get_super_category_dataloaders(super_category)
            super_category_datasizes = self.data.get_super_category_datasizes(super_category)
            # Dataset
            super_category_dataset = self.data.get_super_category_dataset(super_category)
            
            # Model
            model = self.get_model(super_category_dataset)

            # Optimizer and # Scheduler
            if self.is_omniprint_dataset():

                optimizer = optim.Adam(model.parameters(), lr=self.hp.lr)
                scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor = self.hp.factor, patience = self.hp.patience)
            else:
                optimizer = optim.SGD(model.parameters(), lr=self.hp.lr, momentum=self.hp.momentum)
                scheduler = lr_scheduler.StepLR(optimizer, step_size=self.hp.step_size, gamma=self.hp.gamma)
        

        
            
            # Loss
            criterion = self.hp.criterion


            best_train_acc, best_test_acc = 0.0 , 0.0
            loss_history, score_history = [], []
            train_loss, test_loss, train_score, test_score = [], [], [], [] 
            
            
            for epoch in range(self.hp.epochs):
                

                train_labels, test_labels = [], []
                train_predictions, test_predictions = [], []
                train_predicted_probabilities, test_predicted_probabilities = [],[]

                for phase in phases:
                    if phase == 'train':
                        model.train()  # Set model to training mode
                    else:
                        model.eval()   # Set model to evaluate mode
                    
                    running_loss = 0.0
                    running_corrects = 0

                    # Iterate over data.
                    for inputs, labels in super_category_dataloaders[phase]:
                        
                        
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        
                        
                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                        
                            outputs = model(inputs)
                            probabilities = F.softmax(outputs, dim=1)
                            _, preds = torch.max(outputs, 1)


                            loss = criterion(outputs, labels)
                            
                            
                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()

                        # statistics
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)


                        
                        # saving prediction and ground truth for future use
                        if phase == 'train':
                            train_predictions += list(preds.numpy())
                            train_labels += list(labels.numpy())
                            train_predicted_probabilities += list(probabilities.detach().numpy())
                        else:
                            test_predictions += list(preds.numpy())
                            test_labels += list(labels.numpy())
                            test_predicted_probabilities += list(probabilities.detach().numpy())

                    # end dataloader loop

                    if phase == 'train':
                        if not self.is_omniprint_dataset():
                            scheduler.step()
                        
                    else:
                        if self.is_omniprint_dataset():
                            scheduler.step(running_loss)
                    
                    epoch_loss = running_loss / super_category_datasizes[phase]
                    epoch_acc = (running_corrects.double() / super_category_datasizes[phase]).item()

                    

                    loss_history, score_history = (train_loss, train_score) if phase == 'train' else (test_loss, test_score)
                    loss_history.append(epoch_loss)
                    score_history.append(epoch_acc)

                    if phase == 'train' and epoch_acc > best_train_acc:
                        best_train_acc = epoch_acc
                    if phase == 'test' and epoch_acc > best_test_acc:
                        best_test_acc = epoch_acc
                    

                # end phase loop

                self.lprint("Epoch: {0} \tTrain loss: {1} \tTest loss: {2} \tTrain acc: {3} \tTest acc: {4}".format(
                    epoch, "{:.2f}".format(train_loss[-1]), "{:.2f}".format(test_loss[-1]), "{:.2f}".format(train_score[-1]), "{:.2f}".format(test_score[-1])
                ))
                
                

            # end epoch loop

            self.lprint("\nBest Train acc: {0} \tBest Test acc: {1}".format(
                "{:.2f}".format(best_train_acc), "{:.2f}".format(best_test_acc)
            ))
            

            time_elapsed = time.time() - since
    
            training_time = '{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)
            self.lprint("Training completed in: {}\n\n".format(training_time))
            

            # save each category results
            self.save_categories_result(super_category, training_time,
                        best_train_acc, best_test_acc, 
                        train_score, train_loss, test_score, test_loss,
                        train_predictions, train_labels, train_predicted_probabilities,
                        test_predictions, test_labels, test_predicted_probabilities)
        
        # end super-category loop
        





#-----------------------
# Main  
#-----------------------
if __name__ == "__main__":

    # Get args 
    args = parse_arguments()

    # Initiaize training object
    train = Hierarchical_Classification_Training(args=args)

    # set random seeds
    train.set_random_seeds()
        
    # Set device
    train.set_device()
    

    # Train
    train.start_training()
  

    