import torch
import torch.nn as nn
import numpy as np

from .algorithm import Algorithm
from .modules.ownlstm import MetaLearner
from .modules.utils import preprocess_grad_loss, put_on_device, get_loss_and_grads,\
                            accuracy, unflatten_weights

class LSTM(Algorithm):
    """
    Our re-implementation of the meta-learner LSTM of Ravi et al. (2017). 
    Inherits from Algorithm super class.
    
    ...

    Attributes
    ----------
    baselearner_fn : constructor function
        Constructor function for the baselearner
    baselearner_args : dict
        Keyword arguments for base-learner
    loss_fn : loss function
        Loss that ought to be minimized
    opt_fn : constructor function
        Constructor function for the optimizer to use
    T : int
        Number of update steps to parameters per task
    train_batch_size : int
        Indicating the size of minibatches that are sampled from meta-train tasks
    test_batch_size : int
        Size of batches to sample from meta-[val/test] tasks
    lr : float
        Learning rate for the optimizer
    validation : boolean
        Whether the model should use meta-validation
    dev : str
        Device identifier
    trainable : boolean
        Whether it can train on meta-trrain tasks
    grad_clip : float
        Threshold for gradient norm clipping
    input_size : int
        Input size of the first LSTM layer 
    hidden_size : int
        Number of nodes in the hidden state of the LSTM
    learner_w_grad : nn.Module
        Baselearner network for for support set
    learner_wo_grad : nn.Module
        Baselearner which is only used on the query set
    metalearner : nn.Module
        The LSTM meta learning component
    optim : torch.optim
        Optimizer to update the metalearner
    
    Methods
    -------
    train_learner(train_x, train_y)
        Unroll the meta-LSTM on the given task support set
    
    forward_weights(x, weights)
        Perform a feedforward pass on inputs x making using
        @weights instead of the object's weights (w1, w2, w3)
    
    dump_state()
        Dumps the model state
        
    load_state(state)
        Loads the given model state
    """

    def __init__(self, grad_clip, input_size, second_order, hidden_size, **kwargs):
        """
        Parameters
        ----------
        grad_clip : float
            Threshold for gradient norm clipping
        input_size : int
            Input size of the first LSTM layer 
        hidden_size : int
            Number of nodes in the hidden state of the LSTM
        **kwargs : dict
            Keyword arguments passed to the constructor of the super class   
        """

        super().__init__(**kwargs)
        self.grad_clip = grad_clip
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.second_order = second_order

        self.baselearner = self.baselearner_fn(**self.baselearner_args).to(self.dev)
        self.metalearner = MetaLearner(self.input_size, 
                                       self.hidden_size,
                                       self.baselearner.get_flat_params().size(0)).to(self.dev)
        
        self.flat_baselearner_params = torch.cat([p.reshape(-1).clone().detach().to(self.dev) for p in self.baselearner.parameters()], 0)
        self.metalearner.metalstm.init_cI(self.flat_baselearner_params)

        # Set up loss, optimizer, learning rate scheduler
        self.optim = self.opt_fn(self.metalearner.parameters(), self.lr)


    def train_learner(self, train_x, train_y, train_mode=True):
        """Unroll the LSTM on the support set
        
        Apply self.T updates to the baselearner parameters by unrolling
        the LSTM meta learner network
        
        Parameters
        ----------
        train_x : torch.Tensor
            Tensor of inputs
        train_y : torch.Tensor
            Tensor of ground-truth outputs corresponding to the inputs
        train_mode : boolean, optional
            Whether the function is called on a meta-train tasks (default=True)
        second_order : boolean, optional
            Whether to use second-order gradients

        Returns
        ----------
        torch.Tensor
            Resulting (flattened) base-learner parameters  
        """
        if train_mode:
            batch_size = self.train_batch_size
        else:
            batch_size = self.test_batch_size
            
        cI = self.metalearner.metalstm.cI
        hs = [None]
        for _ in range(self.T):
            for i in range(0, len(train_x), batch_size):
                x = train_x[i:i+batch_size]
                y = train_y[i:i+batch_size]

                # get the loss/grad
                self.baselearner.zero_grad()
                unflattened_cI = unflatten_weights(cI, self.baselearner.parameters())

                loss, grad = get_loss_and_grads(self.baselearner, train_x, train_y, 
                                                weights=unflattened_cI, item_loss=False, 
                                                create_graph=self.second_order,
                                                retain_graph=self.T > 1 or self.second_order,
                                                flat=False)
                
                grad = torch.cat([p.data.reshape(-1) / batch_size for p in grad], 0)

                # preprocess grad & loss and metalearner forward
                grad_prep = preprocess_grad_loss(grad)  # [n_learner_params, 2]
                loss_prep = preprocess_grad_loss(loss.data.unsqueeze(0)) # [1, 2]
                metalearner_input = [loss_prep, grad_prep, grad.unsqueeze(1)]
                cI, h = self.metalearner(metalearner_input, hs[-1])
                hs.append(h)
        return cI

    def train(self, train_x, train_y, test_x, test_y):
        """Perform a single training step on the given task
        
        Use the task to perform a single training step. The task consists of:
         support set : train_x, train_y
         query set : test_x, test_y
        Unroll the LSTM for T steps on the support set, compute the loss on
        query set, and backpropagate the loss
        
        Parameters
        ----------
        train_x : torch.Tensor
            Tensor of inputs for the support set
        train_y : torch.Tensor
            Tensor of ground-truth outputs corresponding to the support set inputs
        test_x : torch.Tensor
            Tensor of query set inputs
        test_y : torch.Tensor
            Tensor of query set outputs  
        """
        
        # Put tensors on appropriate devices
        train_x, train_y, test_x, test_y = put_on_device(self.dev,
                                                         [train_x, train_y,
                                                          test_x, test_y])
        
        # Train learner with metalearner
        self.baselearner.reset_batch_stats()
        self.baselearner.train()

        cI = self.train_learner(train_x, train_y)
        unflattened_cI = unflatten_weights(cI, self.baselearner.parameters())

        # Train meta-learner using query set loss
        output = self.baselearner.forward_weights(test_x, unflattened_cI)
        loss = self.baselearner.criterion(output, test_y)

        self.optim.zero_grad()
        loss.backward()
        
        nn.utils.clip_grad_norm_(self.metalearner.parameters(), self.grad_clip)
        self.optim.step()


    def evaluate(self, train_x, train_y, test_x, test_y):   
        """Evaluate the LSTM meta-learner on a single task
        
        Update the base-learner using the support set (train_x, train_y)
        and compute the loss on the query set (test_x, test_y), 
        which is then returned
        
        Parameters
        ----------
        train_x : torch.Tensor
            Tensor of inputs for the support set
        train_y : torch.Tensor
            Tensor of ground-truth outputs corresponding to the support set inputs
        test_x : torch.Tensor
            Tensor of query set inputs
        test_y : torch.Tensor
            Tensor of query set outputs  
        """
        
        # Put tensors on appropriate devices
        train_x, train_y, test_x, test_y = put_on_device(self.dev,
                                                         [train_x, train_y,
                                                          test_x, test_y])
        
        # Train learner with metalearner
        self.baselearner.reset_batch_stats()
        self.baselearner.eval()
        
        cI = self.train_learner(train_x, train_y, train_mode=False)
        unflattened_cI = unflatten_weights(cI, self.baselearner.parameters())
        
        # Train meta-learner using query set loss
        output = self.baselearner.forward_weights(test_x, unflattened_cI)
        loss = self.baselearner.criterion(output, test_y).item() 

        if self.operator == min:
            return loss
        else:
            preds = torch.argmax(output, dim=1)
            return accuracy(preds, test_y)


    def dump_state(self):
        """Return the state of the meta-learner
        
        Returns
        ----------
        state_dict
            State dictionary of the meta-learner
        """
        
        return self.baselearner.state_dict(),\
               self.metalearner.state_dict()
    
    def load_state(self, state):
        """Load the given state into the meta-learner
        
        Parameters
        ----------
        state : state_dict
            State dictionary of the meta-learner
        """
        
        state_base, state_meta = state
        self.baselearner.load_state_dict(state_base)
        self.metalearner.load_state_dict(state_meta)
    