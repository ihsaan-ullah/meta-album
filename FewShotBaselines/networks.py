import pdb
import copy
import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F

# Used to decouple direction and norm of tensors which allow us
# to update only the direction (which is what we want for baseline++)
from torch.nn.utils.weight_norm import WeightNorm


from collections import OrderedDict

class LinearNet(nn.Module):

    def __init__(self, criterion, **kwargs):
        super().__init__()
        self.coeff = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(1))
        self.criterion = criterion
        self.pow = 2

    
    def forward(self, x):
        """Feedforward pass of the network

        Take inputs x, and compute the network's output using its weights
        w1, w2, and w3

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor with shape (Batch size, 1)

        Returns
        ----------
        tensor
            Predictions with shape (Batch size, 1) of the network on inputs x 
        """
        pred = self.coeff*torch.pow(x,self.pow) + self.bias
        return pred


    def forward_weights(self, x, weights):
        """Feedforward pass using provided weights
        
        Take input x, and compute the output of the network with user-defined
        weights
        
        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor with shape (Batch size, 1)
        weights : list
            List of tensors representing the weights of a custom SineNetwork.
            Format: [layer 1 kernel, layer 1 bias, layer 2 kernel,..., layer 3 bias]
        
        Returns
        ----------
        tensor
            Predictions with shape (Batch size, 1) of the implicitly defined network 
            on inputs x 
        """
        
        pred = weights[0]*torch.pow(x,self.pow) + weights[1]
        return pred
    
    def get_flat_params(self):
        """Returns parameters in flat format
        
        Flattens the current weights and returns them
        
        Returns
        ----------
        tensor
            Weight tensor containing all model weights
        """
        return torch.cat([p.view(-1) for p in self.parameters()], 0)

    def copy_flat_params(self, cI): return
    
    def freeze_layers(self): return



class SineNetwork(nn.Module):
    """
    Base-learner neural network for the sine wave regression task.

    ...

    Attributes
    ----------
    model : nn.ModuleDict
        Complete sequential specification of the model
    relu : nn.ReLU
        ReLU function to use after w1 and w2
        
    Methods
    ----------
    forward(x)
        Perform a feed-forward pass using inputs x
    
    forward_weights(x, weights)
        Perform a feedforward pass on inputs x making using
        @weights instead of the object's weights (w1, w2, w3)
    
    get_flat_params()
        Returns all model parameters in a flat tensor
        
    copy_flat_params(cI)
        Set the model parameters equal to cI
        
    transfer_params(learner_w_grad, cI)
        Transfer batch normalizations statistics from another learner to this one
        and set the parameters to cI
        
    freeze_layers()
        Freeze all hidden layers
    
    reset_batch_stats()
        Reset batch normalization stats
    """

    def __init__(self, criterion, in_dim=1, out_dim=1, zero_bias=True, **kwargs):
        """Initializes the model
        
        Parameters
        ----------
        criterion : nn.loss_fn
            Loss function to use
        in_dim : int
            Dimensionality of the input
        out_dim : int
            Dimensionality of the output
        zero_bias : bool, optional
            Whether to initialize biases of linear layers with zeros
            (default is Uniform(-sqrt(k), +sqrt(k)), where 
            k = 1/num_in_features)
        **kwargs : dict, optional
            Trash can for additional arguments. Keep this for constructor call uniformity
        """
        
        super().__init__()
        self.relu = nn.ReLU()
        
        self.model = nn.ModuleDict({'features': nn.Sequential(OrderedDict([
            ('lin1', nn.Linear(in_dim, 40)),
            ('relu1', nn.ReLU()),
            ('lin2', nn.Linear(40, 40)),
            ('relu2', nn.ReLU())]))
        })
        
        # Output layer
        self.model.update({"out": nn.Linear(40, out_dim)})
        self.criterion = criterion
        
        if zero_bias:
            for m in self.model.modules():
                if isinstance(m, nn.Linear):
                    m.bias = nn.Parameter(torch.zeros(m.bias.size()))

    def forward(self, x):
        """Feedforward pass of the network

        Take inputs x, and compute the network's output using its weights
        w1, w2, and w3

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor with shape (Batch size, 1)

        Returns
        ----------
        tensor
            Predictions with shape (Batch size, 1) of the network on inputs x 
        """

        features = self.model.features(x)
        out = self.model.out(features)
        return out
    
    def forward_weights(self, x, weights):
        """Feedforward pass using provided weights
        
        Take input x, and compute the output of the network with user-defined
        weights
        
        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor with shape (Batch size, 1)
        weights : list
            List of tensors representing the weights of a custom SineNetwork.
            Format: [layer 1 kernel, layer 1 bias, layer 2 kernel,..., layer 3 bias]
        
        Returns
        ----------
        tensor
            Predictions with shape (Batch size, 1) of the implicitly defined network 
            on inputs x 
        """
        
        x = F.relu(F.linear(x, weights[0], weights[1]))
        x = F.relu(F.linear(x, weights[2], weights[3]))
        x = F.linear(x, weights[4], weights[5])
        return x
    
    def forward_get_features(self, x):
        features = []
        x = self.model.features.relu1(self.model.features.lin1(x)) # ReLU + Linear
        features.append(x.clone().cpu().detach().numpy())
        x = self.model.features.relu2(self.model.features.lin2(x)) # ReLU + Linear
        features.append(x.clone().cpu().detach().numpy())
        x = self.model.out(x)
        features.append(x.clone().cpu().detach().numpy())
        return x, features

    def forward_weights_get_features(self, x, weights):
        """Manual feedforward pass of the network with provided weights

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor 
        weights : list
            List of torch.Tensor weight variables

        Returns
        ----------
        tensor
            The output of the block
        """
        features = []
        x = F.relu(F.linear(x, weights[0], weights[1])) # ReLU + Linear
        features.append(x.clone().cpu().detach().numpy())
        x = F.relu(F.linear(x, weights[2], weights[3])) # ReLU + Linear
        features.append(x.clone().cpu().detach().numpy())
        x = F.linear(x, weights[4], weights[5])
        features.append(x.clone().cpu().detach().numpy())
        return x, features

    def get_flat_params(self):
        """Returns parameters in flat format
        
        Flattens the current weights and returns them
        
        Returns
        ----------
        tensor
            Weight tensor containing all model weights
        """
        return torch.cat([p.view(-1) for p in self.model.parameters()], 0)

    def copy_flat_params(self, cI):
        """Copy parameters to model 
        
        Set the model parameters to be equal to cI
        
        Parameters
        ----------
        cI : torch.Tensor
            Flat tensor with the same number of elements as weights in the network
        """
        
        idx = 0
        for p in self.model.parameters():
            plen = p.view(-1).size(0)
            p.data.copy_(cI[idx: idx+plen].view_as(p))
            idx += plen

    def transfer_params(self, learner_w_grad, cI):
        """Transfer model parameters
        
        Transfer parameters from cI to this network, while maintaining mean 
        and variance of Batch Normalization 
        
        Parameters
        ----------
        learner_w_grad : nn.Module
            Base-learner network which records gradients
        cI : torch.Tensor
            Flat tensor of weights to copy to this network's parameters
        """
        
        # Use load_state_dict only to copy the running mean/var in batchnorm, the values of the parameters
        #  are going to be replaced by cI
        self.load_state_dict(learner_w_grad.state_dict())
        #  replace nn.Parameters with tensors from cI (NOT nn.Parameters anymore).
        idx = 0
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                wlen = m._parameters['weight'].view(-1).size(0)
                m._parameters['weight'] = cI[idx: idx+wlen].view_as(m._parameters['weight']).clone()
                idx += wlen
                if m._parameters['bias'] is not None:
                    blen = m._parameters['bias'].view(-1).size(0)
                    m._parameters['bias'] = cI[idx: idx+blen].view_as(m._parameters['bias']).clone()
                    idx += blen
    
    def freeze_layers(self):
        """Freeze all hidden layers
        """
        
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.out.weight.requires_grad=True
        self.model.out.bias.requires_grad=True
    
    def reset_batch_stats(self):
        """Resets the Batch Normalization statistics
        """
        
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()
                
class ConvBlock(nn.Module):
    """
    Initialize the convolutional block consisting of:
     - 64 convolutional kernels of size 3x3
     - Batch normalization 
     - ReLU nonlinearity
     - 2x2 MaxPooling layer
     
    ...

    Attributes
    ----------
    cl : nn.Conv2d
        Convolutional layer
    bn : nn.BatchNorm2d
        Batch normalization layer
    relu : nn.ReLU
        ReLU function
    mp : nn.MaxPool2d
        Max pooling layer
    running_mean : torch.Tensor
        Running mean of the batch normalization layer
    running_var : torch.Tensor
        Running variance of the batch normalization layer
        
    Methods
    -------
    forward(x)
        Perform a feed-forward pass using inputs x
    
    forward_weights(x, weights)
        Perform a feedforward pass on inputs x making using
        @weights instead of the object's weights (w1, w2, w3)
    """
    
    def __init__(self, dev, indim=3, pool=True):
        """Initialize the convolutional block
        
        Parameters
        ----------
        indim : int, optional
            Number of input channels (default=3)
        """
        
        super().__init__()
        self.dev = dev
        self.cl = nn.Conv2d(in_channels=indim, out_channels=64,
                            kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(num_features=64, momentum=1) #momentum=1 is crucial! (only statistics for current batch)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(kernel_size=2)
        self.pool = pool
        
    def forward(self, x):
        """Feedforward pass of the network

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor 

        Returns
        ----------
        tensor
            The output of the block
        """

        x = self.cl(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.pool:
            x = self.mp(x)
        return x
    
    def forward_weights(self, x, weights):
        """Manual feedforward pass of the network with provided weights

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor 
        weights : list
            List of torch.Tensor weight variables

        Returns
        ----------
        tensor
            The output of the block
        """
        
        # Apply conv2d
        x = F.conv2d(x, weights[0], weights[1], padding=1) 

        # Manual batch normalization followed by ReLU
        running_mean =  torch.zeros(64).to(self.dev)
        running_var = torch.ones(64).to(self.dev)
        x = F.batch_norm(x, running_mean, running_var, 
                         weights[2], weights[3], momentum=1, training=True)
        if self.pool:                   
            x = F.max_pool2d(F.relu(x), kernel_size=2)
        return x
    
    def reset_batch_stats(self):
        """Reset Batch Normalization stats
        """
        
        self.bn.reset_running_stats()

class ConvX(nn.Module):
    """
    Convolutional neural network consisting of X ConvBlock layers.
     
    ...

    Attributes
    ----------
    model : nn.ModuleDict
        Full sequential specification of the model
    in_features : int
        Number of input features to the final output layer
    train_state : state_dict
        State of the model for training
        
    Methods
    -------
    forward(x)
        Perform a feed-forward pass using inputs x
    
    forward_weights(x, weights)
        Perform a feedforward pass on inputs x making using
        @weights instead of the object's weights (w1, w2, w3)
        
    freeze_layers()
        Freeze all hidden layers
    """
    
    def __init__(self, dev, train_classes, eval_classes, criterion=nn.CrossEntropyLoss(), num_blocks=4, img_size=84):
        """Initialize the conv network

        Parameters
        ----------
        dev : str
            Device to put the model on
        train_classes : int
            Number of training classes
        eval_classes : int
            Number of evaluation classes
        criterion : loss_fn
            Loss function to use (default = cross entropy)
        """
        
        super().__init__()
        self.num_blocks = num_blocks
        self.dev = dev
        self.in_features = 3*3*64
        self.criterion = criterion
        self.train_classes = train_classes
        self.eval_classes = eval_classes

        rnd_input = torch.rand((1,3,img_size,img_size))

        d = OrderedDict([])
        for i in range(self.num_blocks):
            indim = 3 if i == 0 else 64
            pool = i < 4 
            d.update({'conv_block%i'%i: ConvBlock(dev=dev, pool=pool, indim=indim)})
        d.update({'flatten': nn.Flatten()})
        self.model = nn.ModuleDict({"features": nn.Sequential(d)})

        self.in_features = self.get_infeatures(rnd_input).size()[1]
        self.model.update({"out": nn.Linear(in_features=self.in_features,
                                            out_features=self.train_classes).to(dev)})


    def get_infeatures(self,x):
        for i in range(self.num_blocks):
            x = self.model.features[i](x)
        x = self.model.features.flatten(x)
        return x

    def forward(self, x):
        for i in range(self.num_blocks):
            x = self.model.features[i](x)
        x = self.model.features.flatten(x)
        x = self.model.out(x)
        return x

    def forward_weights(self, x, weights, embedding=False):
        """Manual feedforward pass of the network with provided weights

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor 
        weights : list
            List of torch.Tensor weight variables

        Returns
        ----------
        tensor
            The output of the block
        """
        
        for i in range(self.num_blocks):
            x = self.model.features[i].forward_weights(x, weights[i*4:i*4+4])
        x = self.model.features.flatten(x)
        if embedding:
            return x
        x = F.linear(x, weights[-2], weights[-1])
        return x

    def get_flat_params(self):
        """Returns parameters in flat format
        
        Flattens the current weights and returns them
        
        Returns
        ----------
        tensor
            Weight tensor containing all model weights
        """
        return torch.cat([p.view(-1) for p in self.parameters()], 0)

    def copy_flat_params(self, cI):
        """Copy parameters to model 
        
        Set the model parameters to be equal to cI
        
        Parameters
        ----------
        cI : torch.Tensor
            Flat tensor with the same number of elements as weights in the network
        """
        
        idx = 0
        for p in self.parameters():
            plen = p.view(-1).size(0)
            p.data.copy_(cI[idx: idx+plen].view_as(p))
            idx += plen
    
    def load_params(self, state_dict):
        del state_dict["model.out.weight"]
        del state_dict["model.out.bias"]
        self.load_state_dict(state_dict, strict=False)

    def transfer_params(self, learner_w_grad, cI):
        """Transfer model parameters
        
        Transfer parameters from cI to this network, while maintaining mean 
        and variance of Batch Normalization 
        
        Parameters
        ----------
        learner_w_grad : nn.Module
            Base-learner network which records gradients
        cI : torch.Tensor
            Flat tensor of weights to copy to this network's parameters
        """
        
        # Use load_state_dict only to copy the running mean/var in batchnorm, the values of the parameters
        #  are going to be replaced by cI
        self.load_state_dict(learner_w_grad.state_dict())
        #  replace nn.Parameters with tensors from cI (NOT nn.Parameters anymore).
        idx = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                wlen = m._parameters['weight'].view(-1).size(0)
                m._parameters['weight'] = cI[idx: idx+wlen].view_as(m._parameters['weight']).clone()
                idx += wlen
                if m._parameters['bias'] is not None:
                    blen = m._parameters['bias'].view(-1).size(0)
                    m._parameters['bias'] = cI[idx: idx+blen].view_as(m._parameters['bias']).clone()
                    idx += blen
    
    def freeze_layers(self, freeze):
        """Freeze all hidden layers
        """
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.out = nn.Linear(in_features=self.in_features,
                                   out_features=self.eval_classes).to(self.dev)
        self.model.out.bias = nn.Parameter(torch.zeros(self.model.out.bias.size(), device=self.dev))


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, padding, dev):
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
        self.dev = dev
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                               out_channels=out_channels, 
                               kernel_size=3, 
                               stride=stride,
                               padding=padding,
                               bias=False)

        self.bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=1)
        #print("bnsize:", self.bn1.running_mean.size()[0])
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                               out_channels=out_channels, 
                               kernel_size=3, 
                               stride=1,
                               padding=padding,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels, momentum=1)
        self.skip = stride > 1
        if self.skip:
            self.conv3 = nn.Conv2d(in_channels=in_channels, 
                               out_channels=out_channels, 
                               kernel_size=3, 
                               stride=stride,
                               padding=padding,
                               bias=False)
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
        # ResNet doesn't use bias in conv2d layers
        z = F.conv2d(input=x, weight=weights[0], bias=None, 
                     stride=self.stride,padding=self.padding)
        
        z = F.batch_norm(z, torch.zeros(self.bn1.running_mean.size()).to(self.dev), 
                         torch.ones(self.bn1.running_var.size()).to(self.dev), 
                         weights[1], weights[2], momentum=1, training=True)

        z = F.relu(z)

        z = F.conv2d(input=z, weight=weights[3], bias=None, 
                     stride=1, padding=self.padding)

        z = F.batch_norm(z, torch.zeros(self.bn2.running_mean.size()).to(self.dev), 
                         torch.ones(self.bn2.running_var.size()).to(self.dev), weights[4], 
                         weights[5], momentum=1, training=True)

        y = x
        if self.skip:
            y = F.conv2d(input=y, weight=weights[6], bias=None, 
                     stride=self.stride,padding=self.padding)

            y = F.batch_norm(input=y, running_mean=torch.zeros(self.bn3.running_mean.size()).to(self.dev), 
                             running_var=torch.ones(self.bn3.running_var.size()).to(self.dev), weight=weights[7], 
                             bias=weights[8], momentum=1, training=True)

        return F.relu(y + z)

class ResNet(nn.Module):

    def __init__(self, num_blocks, dev, train_classes, eval_classes, criterion=nn.CrossEntropyLoss(), img_size=224):
        super().__init__()
        self.num_blocks = num_blocks
        self.train_classes = train_classes
        self.eval_classes = eval_classes
        self.dev = dev
        self.criterion = criterion
        

        print("ResNet constructor called with num_blocks",num_blocks)
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
            print("Did not recognize the ResNet. It must be resnet10,18,or 34")
            import sys; sys.exit()

        self.num_resunits = sum(layers)

        
        self.conv =  nn.Conv2d(in_channels=3, kernel_size=7, 
                                out_channels=64,
                                stride=2,
                                padding=3,
                                bias=False)
        self.bn =  nn.BatchNorm2d(num_features=64,momentum=1)
        self.relu = nn.ReLU()
        #self.maxpool2d =  nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.globalpool2d = nn.AvgPool2d(7)
        self.flatten = nn.Flatten()
        d = OrderedDict([])

        inpsize = img_size        
        c = 0
        prev_filter = 64
        for idx, (layer, filter) in enumerate(zip(layers, filters)):
            stride = 1
            if idx == 0:
                indim = 64
            else:
                indim = filters[idx-1]
                

            for i in range(layer):
                if i > 0:
                    indim = filter
                if stride == 2:
                    inpsize //= 2
                if prev_filter != filter:
                    stride = 2
                else:
                    stride = 1
                prev_filter = filter


                outsize = int(math.ceil(float(inpsize) / float(stride)))
                if inpsize % stride == 0:
                    padding = math.ceil(max((3 - stride),0)/2)
                else:
                    padding = math.ceil(max(3 - (inpsize % stride),0)/2)


                #padding = math.ceil((3 - stride * 3) * (1 - stride)/2)
                # print("filter:", filter, "input size:", inpsize, "padding:", padding)
                # print("indim:", indim, "stride", stride)
                d.update({'res_block%i'%c: ResidualBlock(in_channels=indim, 
                                                         out_channels=filter,
                                                         stride=stride,
                                                         padding=padding,
                                                         dev=dev)})
                c+=1
        self.model = nn.ModuleDict({"features": nn.Sequential(d)})


        rnd_input = torch.rand((1,3,img_size,img_size))
        self.in_features = self.get_infeatures(rnd_input).size()[1]
        print("Dimensionality of the embedding:", self.in_features)
        self.model.update({"out": nn.Linear(in_features=self.in_features,
                                            out_features=self.train_classes).to(dev)})

        print([x.size() for x in self.model.parameters()])

    def get_infeatures(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        for i in range(self.num_resunits):
            x = self.model.features[i](x)
        # x = F.avg_pool2d(x, kernel_size=7)
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
        #x = F.avg_pool2d(x, kernel_size=7)
        x = F.adaptive_avg_pool2d(x, output_size=(1,1))
        x = self.flatten(x)
        x = self.model.out(x)
        return x

    def forward_weights(self, x, weights, embedding=False):
        z = F.conv2d(input=x, weight=weights[0], bias=None, 
                     stride=2,padding=3)

        z = F.batch_norm(z, torch.zeros(self.bn.running_mean.size()).to(self.dev), 
                         torch.ones(self.bn.running_var.size()).to(self.dev), 
                         weights[1], weights[2], momentum=1, training=True)

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

        #z = F.avg_pool2d(z, kernel_size=7)
        z = F.adaptive_avg_pool2d(z, output_size=(1,1))
        z = self.flatten(z)
        if embedding:
            return z
        z = F.linear(z, weight=weights[-2], bias=weights[-1])
        return z

    def freeze_layers(self, freeze):
        """Freeze all hidden layers
        """
        if freeze:
            for param in self.parameters():
                param.requires_grad = False
        self.model.out = nn.Linear(in_features=self.in_features,
                                   out_features=self.eval_classes).to(self.dev)
        self.model.out.bias = nn.Parameter(torch.zeros(self.model.out.bias.size(), device=self.dev))

    def load_params(self, state_dict):
        del state_dict["model.out.weight"]
        del state_dict["model.out.bias"]
        self.load_state_dict(state_dict, strict=False)