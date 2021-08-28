import torch
import numpy as np

from .algorithm import Algorithm
from .modules.utils import put_on_device, set_weights, get_loss_and_grads,\
                           empty_context, accuracy, deploy_on_task

class MatchingNetwork(Algorithm):
    """Matching networks
    
    Classifies new examples based on point-distance to support examples
    
    ...

    Attributes
    ----------
    baselearner_fn : constructor function
        Constructor function for the base-learner
    baselearner_args : dict
        Dictionary of keyword arguments for the base-learner
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
        Whether this model should use meta-validation
    dev : str
        Device identifier
    trainable : boolean
        Whether it can train on meta-trrain tasks
    episodic : boolean
        Whether to sample tasks or mini batches for training
        
    Methods
    -------
    train(train_x, train_y, test_x, test_y)
        Perform a single training step on a given task
    
    evaluate(train_x, train_y, test_x, test_y)
        Evaluate the performance on the given task
        
    dump_state()
        Dump the meta-learner state s.t. it can be loaded again later
        
    load_state(state)
        Set meta-learner state to provided @state 
    """
    
    def __init__(self, meta_batch_size=1, **kwargs):
        """Initialization of Proto nets
        
        Parameters
        ----------
        meta_batch_size : int
            Number of tasks to compute outer-update
        **kwargs : dict
            Keyword arguments that are ignored
        """
        
        super().__init__(**kwargs)
        self.sine = False
        self.meta_batch_size = meta_batch_size

        # Increment after every train step on a single task, and update
        # init when task_counter % meta_batch_size == 0
        self.task_counter = 0 
        
        # Maintain train loss history
        self.train_losses = []

        # Get random initialization point for baselearner
        self.baselearner = self.baselearner_fn(**self.baselearner_args).to(self.dev)
        self.initialization = [p.clone().detach().to(self.dev) for p in self.baselearner.parameters()]

        # Store gradients across tasks
        self.grad_buffer = [torch.zeros(p.size(), device=self.dev) for p in self.initialization]

        # Enable gradient tracking for the initialization parameters
        for p in self.initialization:
            p.requires_grad = True
                
        # Initialize the meta-optimizer
        self.optimizer = self.opt_fn(self.initialization, lr=self.lr)
        
        # All keys in the base learner model that have nothing to do with batch normalization
        self.keys = [k for k in self.baselearner.state_dict().keys()\
                     if not "running" in k and not "num" in k]
        
        self.bn_keys = [k for k in self.baselearner.state_dict().keys()\
                        if "running" in k or "num" in k]
            

    def _get_params(self):
        return [p.clone().detach() for p in self.initialization]
    
    def _deploy(self, train_x, train_y, test_x, test_y, train_mode):
        """Compute prototypes on support set, compute distances from query inputs to these prototypes
        and predict nearest class' prototype
        
        Parameters
        ----------
        train_x : torch.Tensor
            Inputs of the support set
        train_y : torch.Tensor
            Outputs of the support set
        test_x : torch.Tensor
            Inputs of the query set
        test_y : torch.Tensor
            Outputs of the query set
        train_mode : boolean
            Whether we are in training mode or test mode

        Returns
        ----------
        test_loss
            Loss of the base-learner on the query set after the proposed
            one-step update
        """
        
        # Use torch.no_grad when evaluating -> prevent memory leak
        if not train_mode:
            contxt = torch.no_grad
            num_classes = self.baselearner.eval_classes
        else:
            contxt = empty_context
            num_classes = self.baselearner.train_classes

        with contxt():
            # compute input embeddings
            support_embeddings = self.baselearner.forward_weights(train_x, self.initialization, embedding=True)
            query_embeddings = self.baselearner.forward_weights(test_x, self.initialization, embedding=True)
            
            s_norm = support_embeddings / support_embeddings.norm(dim=1).unsqueeze(1)
            q_norm = query_embeddings / query_embeddings.norm(dim=1).unsqueeze(1)

            # Matrix of cosine similarity scores (i,j)-entry is similarity of 
            # query input i to support example j
            cosine_similarities = torch.mm(s_norm, q_norm.transpose(0,1)).t()

            # Make one-hot encoded matrix of 
            y = torch.zeros((len(train_x), num_classes), device=self.initialization[0].device)
            y[torch.arange(len(train_x)), train_y] = 1

            # cosine_similarities : [n_query, n_support], y : [n_support, n_classes]
            predictions = torch.mm(cosine_similarities, y)

            loss = self.baselearner.criterion(predictions, test_y) 
            
        return loss, predictions
    
    def train(self, train_x, train_y, test_x, test_y):
        """Train on a given task
        
        Start with the common initialization point and perform a few
        steps of gradient descent from there using the support set
        (rain_x, train_y). Observe the error on the query set and 
        propagate the loss backwards to update the initialization.
        
        Parameters
        ----------
        train_x : torch.Tensor
            Inputs of the support set
        train_y : torch.Tensor
            Outputs of the support set
        test_x : torch.Tensor
            Inputs of the query set
        test_y : torch.Tensor
            Outputs of the query set
        """ 
        
        # Put baselearner in training mode
        self.baselearner.train()
        self.task_counter += 1
        # Put all tensors on right device
        train_x, train_y, test_x, test_y = put_on_device(
                                            self.dev,
                                            [train_x, train_y,
                                            test_x, test_y])
        
        # Compute the test loss after a single gradient update on the support set
        test_loss, _ = self._deploy(train_x, train_y, test_x, test_y, True)
        # Propagate the test loss backwards to update the initialization point
        test_loss.backward()
        if self.task_counter % self.meta_batch_size == 0: 
            self.optimizer.step()  
            self.optimizer.zero_grad()

    def evaluate(self, train_x, train_y, test_x, test_y, val=True, compute_cka=False):
        """Evaluate on a given task
        
        Use the support set (train_x, train_y) and/or 
        the query set (test_x, test_y) to evaluate the performance of 
        the model.
        
        Parameters
        ----------
        train_x : torch.Tensor
            Inputs of the support set
        train_y : torch.Tensor
            Outputs of the support set
        test_x : torch.Tensor
            Inputs of the query set
        test_y : torch.Tensor
            Outputs of the query set
        """
        # Put baselearner in evaluation mode
        self.baselearner.eval()
        # Put all tensors on right device
        train_x, train_y, test_x, test_y = put_on_device(
                                            self.dev,
                                            [train_x, train_y,
                                            test_x, test_y])
            
        # Compute the test loss after a single gradient update on the support set
        test_loss, preds = self._deploy(train_x, train_y, test_x, test_y, False)

        if self.operator == min:
            return test_loss.item(), [0]
        else:
            # Turn one-hot predictions into class preds
            preds = torch.argmax(preds, dim=1)
            return accuracy(preds, test_y), [0]
    
    def dump_state(self):
        """Return the state of the meta-learner
        
        Returns
        ----------
        initialization
            Initialization parameters
        """
        return [p.clone().detach() for p in self.initialization]
    
    def load_state(self, state, **kwargs):
        """Load the given state into the meta-learner
        
        Parameters
        ----------
        state : initialization
            Initialization parameters
        """
        
        self.initialization = [p.clone() for p in state]
        for p in self.initialization:
            p.requires_grad = True
        
    def to(self, device):
        self.baselearner = self.baselearner.to(device)
        self.initialization = [p.to(device) for p in self.initialization]
