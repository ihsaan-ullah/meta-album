import torch
import torch.nn as nn
import contextlib
import math
import numpy as np
import pandas as pd

loss_to_init_and_op = {
    nn.MSELoss: (float("inf"), min),
    nn.CrossEntropyLoss: (-float("inf"), max)
}

DATASET_INFO = {
    "resisc": {
        "img_path": "./data/resisc/images/",
        "metadata_path": "./data/resisc/labels.csv",
        "label_col": "category",
        "file_col": "newfilename"
    },
    "insects": {
        "img_path": "./data/insects/images/",
        "metadata_path": "./data/insects/labels.csv",
        "label_col": "CATEGORY",
        "file_col": "FILE_NAME"
    },
    "plankton": {
        "img_path": "./data/plankton/images/",
        "metadata_path": "./data/plankton/labels.csv",
        "label_col": "CATEGORY",
        "file_col": "FILE_NAME"
    },
    "texture1": {
        "img_path": "./data/texture1/images/",
        "metadata_path": "./data/texture1/labels.csv",
        "label_col": "CATEGORY",
        "file_col": "FILE_NAME"
    },
    "texture2": {
        "img_path": "./data/texture2/images/",
        "metadata_path": "./data/texture2/labels.csv",
        "label_col": "CATEGORY",
        "file_col": "FILE_NAME"
    },
    "texture3": {
        "img_path": "./data/texture3/images/",
        "metadata_path": "./data/texture3/labels.csv",
        "label_col": "CATEGORY",
        "file_col": "FILE_NAME"
    },
    "omniprint1": {
        "img_path": "./data/omniprint1/images/",
        "metadata_path": "./data/omniprint1/labels.csv",
        "label_col": "CATEGORY",
        "file_col": "FILE_NAME"
    },
    "omniprint2": {
        "img_path": "./data/omniprint2/images/",
        "metadata_path": "./data/omniprint2/labels.csv",
        "label_col": "CATEGORY",
        "file_col": "FILE_NAME"
    },
    "omniprint3": {
        "img_path": "./data/omniprint3/images/",
        "metadata_path": "./data/omniprint3/labels.csv",
        "label_col": "CATEGORY",
        "file_col": "FILE_NAME"
    },
    "omniprint4": {
        "img_path": "./data/omniprint4/images/",
        "metadata_path": "./data/omniprint4/labels.csv",
        "label_col": "CATEGORY",
        "file_col": "FILE_NAME"
    },
    "dermatology": {
        "img_path": "./data/dermatology/images/",
        "metadata_path": "./data/dermatology/labels.csv",
        "label_col": "label",
        "file_col": "file"
    },
    "rsicb": {
        "img_path": "./data/rsicb/images/",
        "metadata_path": "./data/rsicb/labels.csv",
        "label_col": "category",
        "file_col": "uniquefilename"
    },
    "rsd": {
        "img_path": "./data/rsd/images/",
        "metadata_path": "./data/rsd/labels.csv",
        "label_col": "CATEGORY",
        "file_col": "FILE_NAME"
    },
    "plants": {
        "img_path": "./data/plants/images/",
        "metadata_path": "./data/plants/labels.csv",
        "label_col": "category",
        "file_col": "newfilename"
    },
    "medleaf": {
        "img_path": "./data/medleaf/images/",
        "metadata_path": "./data/medleaf/labels.csv",
        "label_col": "CATEGORY",
        "file_col": "FILE_NAME"
    },
    "flowers": {
        "img_path": "./data/flowers/images/",
        "metadata_path": "./data/flowers/labels.csv",
        "label_col": "CATEGORY",
        "file_col": "FILE_NAME"
    },
}

def process_labels(batch_size, num_classes):
    return torch.arange(num_classes).repeat(batch_size//num_classes).long()

def extract_info(dataset):
    dinfo = DATASET_INFO[dataset]
    return dinfo["label_col"], dinfo["file_col"],\
           dinfo["img_path"], dinfo["metadata_path"]

def train_test_split_classes(classes, test_split=0.3):
    classes = np.unique(np.array(classes))
    np.random.shuffle(classes)
    cut_off = int((1-test_split)*len(classes))
    return classes[:cut_off], classes[cut_off:]

def get_init_score_and_operator(criterion):
    """Get initial score and objective function

    Return the required initialization score and objective function for the given criterion.
    For example, if the criterion is the CrossEntropyLoss, we want to maximize the accuracy.
    Hence, the initial score is set to -infty and we with to maximize (objective function is max)
    In case that the criterion is MSELoss, we want to minimize, and our initial score will be
    +infty and our objective operator min. 
    
    Parameters
    ----------
    criterion : loss_fn
        Loss function used for the base-learner model
    """
    
    for loss_fn in loss_to_init_and_op.keys():
        if isinstance(criterion, loss_fn):
            return loss_to_init_and_op[loss_fn]

def accuracy(y_pred, y):
    """Computes accuracy of predictions
    
    Compute the ratio of correct predictions on the true labels y.
    
    Parameters
    ----------
    y_pred : torch.Tensor
        Tensor of label predictions
    y_pred : torch.Tensor
        Tensor of ground-truth labels 
    
    Returns
    ----------
    accuracy
        Float accuracy score in [0,1]
    """
    
    return ((y_pred == y).float().sum()/len(y)).item()

    
def get_batch(train_x, train_y, batch_size):
    """Sample a minibatch

    Samples a minibatch from train_x, train_y of size batch_size.
    Repetitions are allowed (much faster and only occur with small probability).

    Parameters
    ----------
    train_x : torch.Tensor
        Input tensor
    train_y : torch.Tensor
        Output tensor
    batch_size : int
        Size of the minibatch that should be sampled

    Returns
    ------
    x_batch 
        Tensor of sampled inputs x
    y_batch
        Tensor of sampled outputs y
    """

    batch_indices = np.random.randint(0, train_x.size()[0], batch_size)
    x_batch, y_batch = train_x[batch_indices], train_y[batch_indices]
    return x_batch, y_batch

def update(model, optimizer, train_x, train_y, ret_loss=False):
    """Perform a single model update 

    Apply model on the given data train_x, train_y.
    Compute the prediction loss and make parameter updates using the provided
    optimizer.

    Parameters
    ----------
    model : nn.Module
        Pytorch neural network module. Must have a criterion attribute!!
    optimizer : torch.optim
        Optimizer that updates the model
    train_x : torch.Tensor
        Input tensor x
    train_y : torch.Tensor 
        Ground-truth output tensor y
    ret_loss : bool, optional
        Whether to return the observed loss (default=False)
        
    Returns
    ------
    ret_loss (optional) 
        Only if ret_loss=True
    """
    
    model.zero_grad()
    out = model(train_x)
    loss = model.criterion(out, train_y)
    loss.backward()
    optimizer.step()
    if ret_loss:
        return loss.item()
    
def new_weights(model, best_weights, best_score, train_x, train_y, operator, ls=False):
    """Update the best weights found so far

    Applies model to train_x, train_y and computes the loss.
    If the observed loss is smaller than best_score, the best weights and loss 
    are updated to those of the current model, and the observed loss

    Parameters
    ----------
    model : nn.Module
        Pytorch neural network module. Must have a criterion attribute.
    best_weights : list
        List of best weight tensors so far
    best_score : float
        Best obtained loss so far
    train_x : torch.Tensor
        Input tensor x
    train_y : torch.Tensor 
        Ground-truth output tensor y
    operator : function, optional
        Objective function. In case of RMSE, it is a minimization objective (min function),
        in case of accuracy the maximization objective (max function)
    ls : boolean, optional
        Whether to return best weights as list of tensors (default=False)
        
    Returns
    ------
    best_weights
        Updated best weights (if observed loss smaller)
    best_score
        Updated best loss value (if observed loss smaller)
    """
    
    with torch.no_grad():
        eval_out = model(train_x)
        # RMSE loss
        if operator == min:
            eval_score = model.criterion(eval_out, train_y).item()
        else:
            # Classification ==> accuracy
            preds = torch.argmax(eval_out, dim=1)
            eval_score = accuracy(preds, train_y)
        
        tmp_best = operator(eval_score, best_score)
        # Compute new best score
        if tmp_best != best_score and not math.isnan(tmp_best):
            best_score = tmp_best
            if ls:
                best_weights = [p.clone().detach() for p in model.parameters()]
            else:
                best_weights = model.state_dict()
    return best_weights, best_score

def eval_model(model, x, y, operator):
    """Evaluate the model

    Computes the predictions of model on data x and returns the loss
    obtained by comparing the predictions to the ground-truth y values

    Parameters
    ----------
    model : nn.Module
        Pytorch neural network module. Must have a criterion attribute.
    x : torch.Tensor
        Input tensor x
    y : torch.Tensor 
        Ground-truth output tensor y
    operator : function
        Objective function. In case of RMSE, it is a minimization objective (min function),
        in case of accuracy the maximization objective (max function)
        
    Returns
    ------
    score
        floating point performance value (accuracy or MSE) 
    """
    
    with torch.no_grad():
        out = model(x)
        if operator == min:
            score = model.criterion(out, y).item()
        else:
            # Classification ==> accuracy
            preds = torch.argmax(out, dim=1)
            score = accuracy(preds, y)
    return score

def deploy_on_task(model, optimizer, train_x, train_y, 
                   test_x, test_y, T, test_batch_size, init_score, 
                   operator, cpe=4):
    """Apply non-meta-learning baseline model to the given task

    Use baseline strategy to cope with a task. Train for T epochs on minibatches
    of the support set (train_x, train_y) and keep track of the weights that
    perform best on the entire support set.
    Load these weights after T epochs and evaluate the loss on the test set.

    Make sure the model and tensors are on the same device!

    Parameters
    ----------
    model : nn.Module
        Pytorch base-learner model (with criterion attribute)
    optimizer : torch.optim
        Optimizer to train the base-learner network
    train_x : torch.Tensor
        Tensor of inputs for the support set
    train_y : torch.Tensor
        Tensor of ground-truth outputs corresponding to the support set inputs
    test_x : torch.Tensor
        Tensor of query set inputs
    test_y : torch.Tensor
        Tensor of query set outputs  
    T : int
        Number of epochs to train on minibatches of the support set
    test_batch_size : int
        Size of minibatches to draw from the support set
    init_score : float
        Initial score of the model (e.g., accuracy or RMSE)
    operator : function
        Objective function. In case of RMSE, it is a minimization objective (min function),
        in case of accuracy the maximization objective (max function)
    cpe : int, optional
        Number of times to recompute best weights per episode (default=4)
        
    Returns
    ---------- 
    test_loss
        Floating point performance on the query set. If test_x or test_y is None,
        return None. Performance = accuracy if operator is max, else MSE.
    """
    
    # Best weights and loss so far
    best_weights = model.state_dict() 
    best_score = init_score
    
    # Do cpe number of best weight reconsiderations
    val_after = T // cpe

    # Sample T batches, make updates to the parameters, 
    # and keep track of best weights (measured on entire train set)
    for t in range(T):        
        x_batch, y_batch = get_batch(train_x, train_y, test_batch_size)
        update(model, optimizer, x_batch, y_batch)
        if (t + 1) % val_after == 0:
            best_weights, best_score = new_weights(model, best_weights, best_score, 
                                                  train_x, train_y, operator)

    if not test_x is None and not test_y is None:
        # Set the model weights to the best observed so far and get loss on query set
        model.load_state_dict(best_weights)
        test_score = eval_model(model, test_x, test_y, operator)
        return test_score

def get_loss_and_grads(model, train_x, train_y, flat=True, weights=None, item_loss=True,
                       create_graph=False, retain_graph=False):
    """Computes loss and gradients
    
    Apply model to data (train_x, train_y), compute the loss
    and obtain the gradients.
    
    Parameters
    ----------
    model : nn.Module
        Neural network. We assume the model has a criterion attribute 
    train_x : torch.Tensor
        Training inputs
    train_y : torch.Tensor
        Training targets
    
    Returns
    ----------
    loss
        torch.Tensor of size (#params in model) containing the loss value 
        if flat=True, else a single float
    gradients
        torch.Tensor of size (#params in model) containing gradients w.r.t.
        all model parameters if flat=True, else a structured list
    """
    
    model.zero_grad()
    if weights is None:
        weights = model.parameters()
        out = model(train_x)
    else:
        out = model.forward_weights(train_x, weights)
    
    loss = model.criterion(out, train_y)
    grads = torch.autograd.grad(loss, weights, create_graph=create_graph, 
                                retain_graph=retain_graph)
    
    if flat:
        gradients = torch.cat([p.reshape(-1) for p in grads])
        loss = torch.zeros(gradients.size()).to(train_x.device) + loss.item()
    else:
        gradients = list(grads)
        if item_loss:
            loss = loss.item()
    return loss, gradients

def preprocess_grad_loss(x):
    """Preprocesses gradients or loss
    
    Squeeze gradients/loss in two ways and return the result
    
    Parameters
    ----------
    x : torch.Tensor
        Flattened tensor of gradients

    Returns
    ----------
    torch.Tensor
        Processed gradients tensor of size (#gradients, 2) in case that x=gradients
        else tensor of size (1, 2) in case of x=loss
    """
    
    p = 10
    indicator = (x.abs() >= np.exp(-p)).to(torch.float32)

    # preproc1
    x_proc1 = indicator * torch.log(x.abs() + 1e-8) / p + (1 - indicator) * -1
    # preproc2
    x_proc2 = indicator * torch.sign(x) + (1 - indicator) * np.exp(p) * x
    return torch.stack((x_proc1, x_proc2), 1)

def set_weights(model, params, keys, bn_keys):
    """Set model weights to given parameter values
    
    Parameters
    ----------
    model : nn.Module
        Model 
    params : list
        List of torch.Tensor parameter values to be set
    
    """
                
    sd = model.state_dict()
    new_sd = dict()
    for i, key in enumerate(keys):
        new_sd[key] = params[i]
        
    for key in bn_keys:
        new_sd[key] = sd[key]

    model.load_state_dict(new_sd)

def put_on_device(dev, tensors):
    """Put arguments on specific device

    Places the positional arguments onto the user-specified device

    Parameters
    ----------
    dev : str
        Device identifier
    tensors : sequence/list
        Sequence of torch.Tensor variables that are to be put on the device
    """
    for i in range(len(tensors)):
        if not tensors[i] is None:
            tensors[i] = tensors[i].to(dev)
    return tensors

def get_params(model, dev):
    """Get parameters of the model (ignoring batchnorm)

    Retrieves all parameters of a given model and computes slices 

    Parameters
    ----------
    model : nn.Module
        Pytorch model from which we extract the parameters
    dev : str
        Device identifier to place the parameters on

    Returns
    ----------
    params
        List of parameter tensors
    slices
        List of tuples (lowerbound, upperbound) that delimit layer parameters
        E.g., if model has 2 layers with 50 and 60 params, the slices will be
        [(0,50), (50,110)]
    """

    params = []
    slices = []

    lb = 0
    ub = 0
    for m in model.modules():
        # Ignore batch-norm layers
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
            # All parameters in a given layer
            mparams = m.parameters()
            sizes = []
            for mp in mparams:
                sizes.append(mp.numel())
                params.append(mp.clone().detach().to(dev)) 
            # Compute the number of parameters in the layer
            size = sum(sizes)
            # Compute slice indices of the given layer 
            ub += size
            slices.append(tuple([lb, ub]))
            lb += size
    return params, slices

def unflatten_weights(flat_weights, shaped_weights):
    lb = 0
    ub = 0
    unflattened = []
    for tensor in shaped_weights:
        num_els = tensor.numel()
        ub += num_els
        unflattened.append( flat_weights[lb:ub].reshape(tensor.size()) )
        lb += num_els
    return unflattened

# Used as empty context to allow for a conditional `with:' statement
@contextlib.contextmanager
def empty_context():
    yield None
        