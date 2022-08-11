import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, padding, dev):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.dev = dev

        self.conv1 = nn.Conv2d(in_channels=in_channels, 
            out_channels=out_channels, kernel_size=3, stride=stride,
            padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
            out_channels=out_channels, kernel_size=3, stride=1, 
            padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels, momentum=1)
        self.skip = stride > 1
        if self.skip:
            self.conv3 = nn.Conv2d(in_channels=in_channels, 
                out_channels=out_channels, kernel_size=3, stride=stride,
                padding=padding, bias=False)
            self.bn3 = nn.BatchNorm2d(num_features=out_channels, momentum=1)


    def forward(self, x):
        z = self.conv1(x)
        z = self.bn1(z)
        z = self.relu(z)
        z = self.conv2(z)
        z = self.bn2(z)

        y = x
        if self.skip:
            y = self.conv3(y)
            y = self.bn3(y)
        return self.relu(y + z)


    def forward_weights(self, x, weights):
        z = F.conv2d(input=x, weight=weights[0], bias=None, stride=self.stride,
            padding=self.padding)
        
        z = F.batch_norm(z, 
            torch.zeros(self.bn1.running_mean.size()).to(self.dev), 
            torch.ones(self.bn1.running_var.size()).to(self.dev), weights[1], 
            weights[2], momentum=1, training=True)

        z = F.relu(z)

        z = F.conv2d(input=z, weight=weights[3], bias=None, stride=1, 
            padding=self.padding)

        z = F.batch_norm(z, 
            torch.zeros(self.bn2.running_mean.size()).to(self.dev), 
            torch.ones(self.bn2.running_var.size()).to(self.dev), weights[4],
            weights[5], momentum=1, training=True)

        y = x
        if self.skip:
            y = F.conv2d(input=y, weight=weights[6], bias=None, 
                stride=self.stride, padding=self.padding)

            y = F.batch_norm(y, 
                torch.zeros(self.bn3.running_mean.size()).to(self.dev), 
                torch.ones(self.bn3.running_var.size()).to(self.dev), 
                weights[7], weights[8], momentum=1, training=True)

        return F.relu(y + z)


class ResNet(nn.Module):

    def __init__(self, num_blocks, dev, train_classes, eval_classes= None,
                 criterion=nn.CrossEntropyLoss(), img_size=128):
        super().__init__()
        self.num_blocks = num_blocks
        self.train_classes = train_classes
        self.eval_classes = eval_classes
        self.dev = dev
        self.criterion = criterion

        if num_blocks == 10:
            layers = [1,1,1,1]
            filters = [64,128,256,512]
        elif num_blocks == 18:
            layers = [2,2,2,2]
            filters = [64,128,256,512]
        elif num_blocks == 34:
            layers = [3,4,6,3]
            filters = [64,128,256,512]
        else:
            print("ResNet not recognize. It must be resnet10, 18, or 34")
            import sys; sys.exit()

        self.num_resunits = sum(layers)

        
        self.conv =  nn.Conv2d(in_channels=3, kernel_size=7, out_channels=64,
            stride=2, padding=3, bias=False)
        self.bn =  nn.BatchNorm2d(num_features=64,momentum=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        
        d = OrderedDict([])

        inpsize = img_size        
        c = 0
        prev_filter = 64
        for idx, (layer, filter) in enumerate(zip(layers, filters)):
            stride = 1
            if idx == 0:
                in_channels = 64
            else:
                in_channels = filters[idx-1]
                
            for i in range(layer):
                if i > 0:
                    in_channels = filter
                if stride == 2:
                    inpsize //= 2
                if prev_filter != filter:
                    stride = 2
                else:
                    stride = 1
                prev_filter = filter


                if inpsize % stride == 0:
                    padding = math.ceil(max((3 - stride), 0)/2)
                else:
                    padding = math.ceil(max(3 - (inpsize % stride),0)/2)


                d.update({f"res_block{c}": ResidualBlock(
                    in_channels=in_channels, out_channels=filter,
                    stride=stride, padding=padding, dev=dev)})
                c+=1
        self.model = nn.ModuleDict({"features": nn.Sequential(d)})

        rnd_input = torch.rand((1,3,img_size,img_size))
        self.in_features = self.compute_in_features(rnd_input).size()[1]
        self.model.update({"out": nn.Linear(in_features=self.in_features,
            out_features=self.train_classes).to(dev)})

    def compute_in_features(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        for i in range(self.num_resunits):
            x = self.model.features[i](x)
        x = F.adaptive_avg_pool2d(x, output_size=(1,1))
        x = self.flatten(x)
        return x

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        for i in range(self.num_resunits):
            x = self.model.features[i](x)
        x = F.adaptive_avg_pool2d(x, output_size=(1,1))
        x = self.flatten(x)
        x = self.model.out(x)
        return x

    def forward_weights(self, x, weights, embedding=False):
        z = F.conv2d(input=x, weight=weights[0], bias=None, stride=2,
            padding=3)

        z = F.batch_norm(z, 
            torch.zeros(self.bn.running_mean.size()).to(self.dev), 
            torch.ones(self.bn.running_var.size()).to(self.dev), weights[1], 
            weights[2], momentum=1, training=True)

        z = F.relu(z)
        z = F.max_pool2d(z, kernel_size=3, stride=2, padding=1)

        lb = 3
        for i in range(self.num_resunits):
            if self.model.features[i].skip:
                incr = 9
            else:
                incr = 6
            z = self.model.features[i].forward_weights(z, weights[lb:lb+incr])
            lb += incr

        z = F.adaptive_avg_pool2d(z, output_size=(1,1))
        z = self.flatten(z)
        if embedding:
            return z
        z = F.linear(z, weight=weights[-2], bias=weights[-1])
        return z

    def modify_out_layer(self, num_classes):
        if num_classes is None:
            num_classes = self.eval_classes
        self.model.out = nn.Linear(in_features=self.in_features,
            out_features=num_classes).to(self.dev)
        self.model.out.bias = nn.Parameter(torch.zeros(
            self.model.out.bias.size(), device=self.dev))
        
    def freeze_layers(self, freeze, num_classes):
        if freeze:
            for param in self.parameters():
                param.requires_grad = False
        self.modify_out_layer(num_classes)

    def load_params(self, state_dict):
        del state_dict["model.out.weight"]
        del state_dict["model.out.bias"]
        self.load_state_dict(state_dict, strict=False)
