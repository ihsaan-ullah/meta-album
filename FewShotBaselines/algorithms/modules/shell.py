import enum
import torch
import torch.nn as nn

from collections import OrderedDict
from .utils import get_loss_and_grads, preprocess_grad_loss

class Input(enum.Enum):
    RawGrads = 0 # raw gradients
    RawLossGrads = 1 # raw gradients + loss
    ProcGrads = 2 # processed gradients
    ProcLossGrads = 3 # Processed gradients + processed loss

class History(enum.Enum):
    Empty = 0
    Gradients = 1
    Updates = 2

TYPE_TO_SIZE = {"raw_grads": 1,
                "raw_loss_grads": 2,
                "proc_grads": 2,
                "proc_loss_grads": 4,
                "maml": 1}

HISTORY_TO_SIZE = {
    "none": 0,
    "grads": 1,
    "updates": 1
}

INP_TO_CODE = {
    "maml": Input.RawGrads,
    "raw_grads": Input.RawGrads,
    "raw_loss_grads": Input.RawLossGrads,
    "proc_grads": Input.ProcGrads,
    "proc_loss_grads": Input.ProcLossGrads
}

HIST_TO_CODE = {
    "none": History.Empty,
    "grads": History.Gradients,
    "updates": History.Updates
}

class MetaLearner(nn.Module):
    """
    Meta-learner class. This is the meta network that proposes updates
    for given processed loss and gradient values.
    
    ...

    Attributes
    ----------
    w1 : nn.Linear
        First dense layer of the neural network
    w2 : nn.Linear
        Second dense layer 
    w3: nn.Linear
        Output layer
    activation : act_fn
        Activation function to use in all hidden layers
    criterion : loss_fn
        Loss function to use
    
    Methods
    ----------
    forward(x)
        Perform a single feed-forward pass of the network, resulting
        in a proposed update
    
    """
    
    def __init__(self, input_type, activation, layers, history="none", time_input=False):
        """Initialize the network
        
        Initialize all layers of the network:
         - 4 input nodes
         - 20 nodes in layer 1
         - 20 nodes in layer 2
         - 1 node in output layer
         Set all biases to zero
         
         Arguments
         ----------
         input_type : str
             Specifies the input of the OSO network (raw_grads, raw_loss_grads,
             proc_loss_grads, maml)
         activation : act_fn
             The activation function to use in all hidden layers
         layers : list
             List of number of neurons in hidden layers (e.g., [4,10,1] is a neural network
             with 4 hidden nodes in layer 1 and 10 in layer 2 and 1 output node
        history : str, optional
            Type of historical information to use [none, grads, updates]
        time_input : boolean, optional
            Whether to add a timestamp as input
        """
        
        super().__init__()
        self.criterion = nn.MSELoss()
        self.activation = activation
        
        if input_type == "maml":
            assert len(layers) == 1, "Input type is maml. Incompatible with multiple OSO layers."
            assert layers[0] == 1, f"Input type is maml. Incompatible with {layers[0]} input nodes"
        
        info_size = TYPE_TO_SIZE[input_type]
        history_size = HISTORY_TO_SIZE[history]
        # If we have to process the history, the size is doubled
        if "proc" in input_type:
            history_size *= 2
        
        input_size = info_size + history_size

        if time_input:
            input_size += 1

        print("Input size:", input_size)

        # Create feature extraction module (body of the network)
        name = "lin"
        actname = "act"
        tuples = []
                
        # Example of layers = [20,20,1]
        for i, block_size in enumerate(layers[:-1]):
            # Create new dense layer
            linear_block = nn.Linear(input_size, block_size)
            # Set biases to 0
            linear_block.bias = nn.Parameter(torch.zeros(linear_block.bias.size()))
            # Add dense layer to blocks
            tuples.append(tuple([name+str(i+1),linear_block]))
            tuples.append(tuple([actname+str(i+1), activation]))
            # Set input size of next layer to output size of current layer
            input_size = block_size
        
        self.model = nn.ModuleDict({'features': nn.Sequential(OrderedDict(tuples))})
        
        # Add head of the network (proposed weight update)
        self.model.update({"out": nn.Linear(input_size, layers[-1])})
        self.model.out.bias = nn.Parameter(torch.zeros([1]))
        # If input type is maml, use inner LR used by MAML (0.01) 
        if input_type == "maml":
            self.model.out.weight = nn.Parameter(torch.ones([1,1]) * -0.01)

        print("[*] OSO network created:\n", self.model)
            
    def forward(self, x):
        """Perform a feed-forward pass
        
        Take inputs x and propose a weight update
        
        Returns
        ----------
        output
            Proposed weight update as torch.Tensor
        """
        
        x = self.model.features(x)
        x = self.model.out(x)
        return x

def get_init_info(baselearner, train_x, train_y, weights=None, 
                  create_graph=False, retain_graph=False, input_code=None, grad_clip=None, meta_loss=False, class_map=None, loss_net=None, loss_params=None):
    """Compute initial loss tensor and gradients

    Compute the loss on the given support set (train_x, train_y)
    and gradients w.r.t. all baselearner parameters. 
    Preprocess these values to construct meta input matrix X.

    Parameters
    ----------
    baselearner : nn.Module
        Baselearner model
    train_x : torch.Tensor
        Inputs of the support set
    train_y : torch.Tensor
        Outputs of the support set
    weights : list, optional
        List of parameter tensors which should be used in the forward pass 
        (default=None)
    create_graph : boolean, optional
        Whether to create a tensor graph (only used to compute second-order gradients)
        (default=False)
    retain_graph : boolean, optional
        Whether to keep the computational graph (only used for second-order gradients)
        (default=False)
    input_code : int
        Specifier of the input type 
           - Input.RawGrads : raw gradients
           - Input.RawLossGrads : raw loss concatenated with gradients
           - Input.ProcGrads : processed gradients
           - Input.ProcLossGrads : processed loss concatenated with processed gradients

    Returns
    ----------
    X
        torch.Tensor of meta-inputs. Shape = (#params in baselearner, 4)
    """

    # Compute loss tensor and gradients of shape
    # (num params in baselearner)
    init_losses, init_grads = get_loss_and_grads(baselearner, 
                                                 train_x, train_y,
                                                 weights=weights,
                                                 create_graph=create_graph,
                                                 retain_graph=retain_graph,
                                                 meta_loss=meta_loss, 
                                                 class_map=class_map,
                                                 loss_net=loss_net,
                                                 loss_params=loss_params)
    
    if not grad_clip is None:
        init_grads = torch.clamp(init_grads, -grad_clip, +grad_clip)


    if input_code == Input.RawGrads:
        return torch.reshape(init_grads, [-1, 1]), torch.reshape(init_grads, [-1, 1])
    elif input_code == Input.RawLossGrads:
        return torch.stack((init_losses, init_grads), 1), torch.reshape(init_grads, [-1, 1])
    
    # Else, we must preprocess the losses and gradients
    # Preprocess these initial values according to 
    # Andrychowicz et al. (2016) and Ravi et al. (2017)
    init_grads = preprocess_grad_loss(init_grads)
    if input_code == Input.ProcGrads:
        return init_grads, torch.reshape(init_grads, [-1, 2])

    init_losses = preprocess_grad_loss(init_losses)
    # Concatenate these inital values (column-wise) 
    # to create meta-input
    return torch.cat((init_losses, init_grads), 1), torch.reshape(init_grads, [-1, 2])

def get_fast_weights(updates, initialization):
    """Compute task-specific weights
    
    Takes initialization parameters and task-specific weight updates 
    and creates the corresponding, new (fast) weights.
    
    Parameters
    ----------
    updates : torch.Tensor
        Updates proposed for parameters. Shape = (#params,1)
    initialization : list
        List of initialization parameters. Every element in the list
        is a torch.Tensor.

    Returns
    ----------
    fast_weights
        Task-specific weights obtained by computing initialization + updates

    """
    
    fast_weights = []
    
    lb = 0
    ub = 0
    for lid, l in enumerate(initialization):
        num_els = l.numel()
        ub += num_els
        fast_weights.append(l + updates[lb:ub].reshape(l.size()))
        lb += num_els
    return fast_weights