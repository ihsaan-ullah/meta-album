import sys
import os

# in order to import modules from packages in the parent directory
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
from torchvision import models



# Custom ResNet-18 class
class ResNet18:

    

    # Constructor's Arguments:
    # pre_trained : Boolean - to intiialize a model with image-net weights.
    # only_train_last_layer : Boolean - to freeze all layers except last layer
    # number_of_classes : Integer - to fix the output size of FC layer.
    # device : CPU/GPU
    def __init__(self, pre_trained=True, only_train_last_layer=True, number_of_classes=2, device=None):

        model = models.resnet18(pretrained=pre_trained)
        self.device = device
        if only_train_last_layer:
            for param in model.parameters():
                param.requires_grad = False
        
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, number_of_classes)
    
        self.model= model
        self.model_name = "ResNet18"


    # Returns the model
    def get_model(self):
        return self.model.to(self.device)

    def get_name(self):
        return self.model_name
