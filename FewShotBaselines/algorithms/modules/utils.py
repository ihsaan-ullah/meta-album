import torch
import torch.nn as nn
import contextlib
import math
import numpy as np

loss_to_init_and_op = {
    nn.MSELoss: (float("inf"), min),
    nn.CrossEntropyLoss: (-float("inf"), max)
}

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

def update(model, optimizer, train_x, train_y, ret_loss=False, get_acc=False):
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
    
    optimizer.zero_grad()
    out = model(train_x)
    preds = torch.argmax(out, dim=1)
    acc_score = accuracy(preds, train_y)
    loss = model.criterion(out, train_y)
    loss.backward()
    optimizer.step()
    if ret_loss:
        return loss.item()
    if get_acc:
        return acc_score
    
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
            acc = None
            loss = model.criterion(out, y).item()
        else:
            # Classification ==> accuracy
            preds = torch.argmax(out, dim=1)
            acc = accuracy(preds, y)
            loss = model.criterion(out, y).item()
    return acc, loss

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


    loss_history = []

    # Sample T batches, make updates to the parameters, 
    # and keep track of best weights (measured on entire train set)
    for t in range(T):        
        x_batch, y_batch = get_batch(train_x, train_y, test_batch_size)
        lossval = update(model, optimizer, x_batch, y_batch, ret_loss=True)
        loss_history.append(lossval)
        if (t + 1) % val_after == 0:
            best_weights, best_score = new_weights(model, best_weights, best_score, 
                                                  train_x, train_y, operator)

    # if val_after > T-1, best_weights = current_weights (because state_dict holds a reference).
    # thus the code still works
    if not test_x is None and not test_y is None:
        # Set the model weights to the best observed so far and get loss on query set
        model.load_state_dict(best_weights)
        acc, loss = eval_model(model, test_x, test_y, operator)
        loss_history.append(loss)
        return acc, loss_history



def process_cross_entropy(preds, targets, class_map, apply_softmax, dev, log=False, single_input=False):
    """Converts the predictions and targets into a format that CrossEntropy expects
    Every row of the new input will consist of [one-hot encoding for preds, one-hot encoding for target]
    
    Args:
        preds (torch.Tensor): tensor of predictiopns of shape [num_examples,]
        targets (torch.Tensor): ground-truth labels of shape [num_examples,]
        class_map (dict): maps classes to column positions (ints) in the one-hot encoding 
        apply_softmax (bool): whether to apply the softmax to the predictions
        dev (str): device identifier to put the inputs on 
        log (bool): whether to take the log of inputs

    Returns:
        torch.Tensor: tensor of one-hot encoded predictions and targets of shape [num_examples, in_dim]
    """
    
    one_hot = torch.zeros((preds.size(0), 2*len(class_map.keys())), device=dev)
    # this is the case of binary classification (only 1 output node, but 2 classes)
    if len(class_map.keys()) == 2:
        class_a, class_b = list(class_map.keys())
        # do the predictions
        one_hot[:, 0] = preds.view(-1)
        one_hot[:, 1] = 1 - preds.view(-1)
        if apply_softmax:
            one_hot[:,:2] = torch.softmax(one_hot[:,:2].clone(), dim=1)
        one_hot[targets == class_a, 2] = 1
        one_hot[targets == class_b, 3] = 1
        if log and not single_input:
            one_hot = torch.log(one_hot + 1e-5)

        outputs = one_hot[:,2].detach().float().view(-1,1)
        if single_input:
            if not log:
                one_hot = (one_hot[:,:2] * one_hot[:,2:]).sum(dim=1).unsqueeze(1)
            else:
                one_hot = torch.log((one_hot[:,:2] * one_hot[:,2:]).sum(dim=1).unsqueeze(1))


    else:
        outputs = torch.zeros(targets.size(), dtype=torch.long, device=dev)
        num_classes = len(class_map.keys())
        for c, column in class_map.items():
            column = class_map[c]
            one_hot[:, column] = preds[:, column]
            one_hot[targets == c, num_classes + column] = 1 
            outputs[targets == c] = column
        if apply_softmax:
            one_hot[:,:num_classes] = torch.softmax(one_hot[:,:num_classes].clone(), dim=1)
        if log and not single_input:
            one_hot = torch.log(one_hot + 1e-5)
        if single_input:
            if not log:
                one_hot = (one_hot[:,:num_classes] * one_hot[:,num_classes:]).sum(dim=1).unsqueeze(1)
            else:
                one_hot = torch.log((one_hot[:,:num_classes] * one_hot[:,num_classes:]).sum(dim=1).unsqueeze(1))

    return one_hot, outputs


def get_loss_and_grads(model, train_x, train_y, flat=True, weights=None, item_loss=True,
                       create_graph=False, retain_graph=False, rt_only_loss=False, 
                       meta_loss=False, class_map=None, loss_net=None, loss_params=None):
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
    
    if not meta_loss:
        loss = model.criterion(out, train_y)
    else:
        meta_inputs, targets = process_cross_entropy(out, train_y, class_map=class_map, apply_softmax=True, dev=model.dev)
        loss = loss_net(meta_inputs, weights=loss_params)
    
    if rt_only_loss:
        return loss, None
    
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
        