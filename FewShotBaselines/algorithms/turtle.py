import torch
import numpy as np

from .algorithm import Algorithm
from .modules.shell import MetaLearner, get_init_info, get_fast_weights,\
                           INP_TO_CODE, HIST_TO_CODE, Input, History
from .modules.utils import set_weights, put_on_device, empty_context,\
                           get_init_score_and_operator, accuracy, get_params

class Turtle(Algorithm):
    """sTateless neURal meTa LEarning (TURTLE)
    
    Meta-learning algorithm that combines the best of MAML and 
    LSTM meta-learners. It learns both a good common initialization 
    point for tasks, and learns how to make parameter updates in a
    using a meta-learner network.
    
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
    
    def __init__(self, input_type, history, beta, grad_clip, layers, act, 
                 second_order, layer_wise, param_lr, decouple, meta_batch_size=1, time_input=False, **kwargs):
        """Set all (inherited) attributes
        
        Attributes
        ----------
        input_type : str
            Specifier of the input type (raw_grads, raw_loss_grads, proc_loss_grads,
            maml). Determines the input to the meta-learner network
        layers : list
            List of integers corresponding to the number of neurons per hidden/output layer
            e.g., [5,5,1] is a neural network [input -> hidden 5 -> hidden 5 -> output 1]
        history : str
            Type of historical information to use [none, grads, updates]
        beta : float
            Float between [0,1] that determines the memory capacity for the historical
            information. Only affects results if history != none
        grad_clip : float
            Gradient clipping value to use
        act : act_fn
            Activation function to user for meta-learner
        second_order : boolean
            Whether to use second-order gradient information
        layer_wise : boolean
            Whether to use a meta-learner network for every individual base-learner layer
        meta_batch_size : int
            Number of tasks to compute outer-update
        time_input : boolean
            Whether to add a timestamp to the input (this way TURTLE can find an equivalent of a learning schedule)
        param_lr : boolean
            Whether to maintain a learning rate per parameter
        decouple : int
            Whether to decouple the base- and meta-search processes (int indicates the task # to switch modes)
        **kwargs : dict
            Ignored keyword arguments
        """
        
        super().__init__(**kwargs)
        self.input_type = input_type
        self.input_code = INP_TO_CODE[input_type]
        self.history_code = HIST_TO_CODE[history]
        self.beta = beta
        self.grad_clip = grad_clip
        self.second_order = second_order
        self.layer_wise = layer_wise
        self.meta_batch_size = meta_batch_size
        self.time_input = time_input
        self.param_lr = param_lr
        self.decouple = decouple
        self.metasearch = bool(self.decouple) # currently looking only for optimizer if self.decouple
        self.dcounter = 0
        # Increment after every train step on a single task, and update
        # init when task_counter % meta_batch_size == 0
        self.task_counter = 0 

        # Get random initialization point for baselearner
        self.baselearner = self.baselearner_fn(**self.baselearner_args).to(self.dev)
        self.initialization, self.slices = get_params(self.baselearner, self.dev)

        # If we do not use layer-wise meta-learner networks, there is only a single parameter slice
        # namely (0, num_params)
        if not self.layer_wise:
            self.slices = [ tuple([0, self.slices[-1][1]]) ]
            print(self.slices) 

        self.num_params = 0
        # Enable gradient tracking for the initialization parameters    
        for p in self.initialization:
            self.num_params += p.numel()
            p.requires_grad = True

        # We do not use historical information when T = 1
        if self.T == 1:
            ml_hist = "none" 
        else:
            ml_hist = history

        # Create meta-learners for every layer of the base-learner network 
        # (if layerwise is true, else we have one meta-learner network for the entire base-learner)
        self.metalearners = [MetaLearner(input_type=input_type, layers=layers, 
                                activation=act, history=ml_hist, time_input=time_input).to(self.dev) for layer in self.slices]


        # If input type is 'maml', the meta-learner network parametesr are fixed
        self.meta_params = list(self.initialization) if not self.decouple else []
        if not input_type == "maml":
            for ml in self.metalearners:
                self.meta_params += list(ml.parameters())
        
        # If we need to learn a learning rate per parameter
        if self.param_lr:
            self.lr_vector = torch.rand(size=(self.num_params, 1), device=self.dev).detach()
            self.lr_vector.requires_grad=True

            self.meta_params += [self.lr_vector]

        # Create gradient buffers that keep track of gradients across tasks
        self.grad_buffer = [torch.zeros(p.size(), device=self.dev) for p in self.meta_params]

        self.optimizer = self.opt_fn(self.meta_params, lr=self.lr)
        
        # All keys in the base learner model that have nothing to do with batch normalization
        self.keys = [k for k in self.baselearner.state_dict().keys()\
                     if not "running" in k and not "num" in k]
        
        self.bn_keys = [k for k in self.baselearner.state_dict().keys()\
                        if "running" in k or "num" in k]
        
    def _get_grad_buffers(self):
        """Initialize empty gradient buffers

        Create gradient buffers to store gradients across tasks.
        This allows for making outer-updates across batches of tasks.
        """

        # Create gradient buffers that keep track of gradients across tasks
        self.init_grad_buffer = [torch.zeros(p.size()) for p in self.initialization]
        self.meta_grad_buffers = []
        for meta_learner in self.metalearners:
            buffer = [torch.zeros(p.size()) for p in meta_learner.parameters()]
            self.meta_grad_buffers.append(buffer)
        
    
    def _history(self, H, step, grads, updates):
        """ Compute historical information tensor

        Obtain the appropriate historical information in tensor format that can be
        concatenated to the meta-learner input X. 

        Parameters
        ----------
        H : torch.Tensor
            Current historical information tensor
        step : int
            Step index (of inner-optimization procedure)
        grads : torch.Tensor
            Gradient tensor of loss w.r.t. all base-learner parameters
        updates : torch.Tensor
            Updates that were made in the previous step (initialized with 0 if no step has been made yet)
            
        Returns
        ----------
        torch.Tensor
            Updated tensor of historical information which can added to the meta-input tensor X
        """
        
        # If in the first step, we have to initialize the history
        if step == 0:
            # If our history type is gradients, initialize with the gradients
            if self.history_code == History.Gradients:
                H = grads
                return H
            # In case of updates, initialize with zeros because we didn't make any updates
            elif self.history_code == History.Updates:
                return torch.zeros(grads.size(), device=self.dev)

        # Initialize the updates with previously made update
        if step == 1 and self.history_code == History.Updates:
            return updates
        
        # Else, the step >= 1 H is no longer empty ==> simply update
        if self.history_code == History.Updates:
            new = updates
        else:
            new = grads

        new_H = self.beta * H + (1 - self.beta) * new
        return new_H

    def _deploy(self, train_x, train_y, test_x, test_y):
        """Run TURTLE on a single task to get the loss on the query set
        
        1. Evaluate the base-learner loss and gradients on the support set (train_x, train_y)
        using our initialization point.
        2. Make a single weight update based on this information.
        3. Evaluate and return the loss of the fast weights (initialization + proposed updates)
        
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
            
        Returns
        ----------
        test_loss
            Loss of the base-learner on the query set after the proposed
            one-step update
        """

        H = None # Empty history
        updates = None # Empty updates
        fast_weights = [p.clone() for p in self.initialization]

        for step in range(self.T):
            # Construct meta input matrix
            # set_weights(self.baselearner, fast_weights, self.keys, self.bn_keys)
            X, grads = get_init_info(self.baselearner, train_x, train_y, weights=fast_weights,
                                 create_graph=self.second_order, 
                                 retain_graph=self.T > 1 or self.second_order,
                                 input_code=self.input_code, grad_clip=self.grad_clip)

            # If we use historical information
            if self.history_code != History.Empty:
                H = self._history(H, step, grads, updates)
                X = torch.cat((X, H), 1)

            # If time should be part of input, concatenate a column of <step> 
            # to X
            if self.time_input:
                col = torch.ones(len(X), 1, device=self.dev) * step
                X = torch.cat((X, col), 1)

            X_slices = [X[lb:ub] for (lb, ub) in self.slices]

            updates = None
            # Compute weight update proposals 
            # Slice the meta-input matrix X into pieces: one for every layer of the base-learner
            # (if layerwise is True, else we use one slice for the entire base-learner network)
            for lid, (lb, ub) in enumerate(self.slices):
                X_slice = X[lb:ub]
                if updates is None:
                    if not self.param_lr:
                        updates = self.metalearners[lid](X_slice)
                    else:
                        updates = torch.mul(self.metalearners[lid](X_slice), self.lr_vector[lb:ub])
                else:
                    if not self.param_lr:
                        updates = torch.cat([updates, self.metalearners[lid](X_slice)])
                    else:
                        updates = torch.cat([updates, torch.mul(self.metalearners[lid](X_slice), self.lr_vector[lb:ub])])
                    

            # Compute the task-specific weights by adding the updates
            # to our initialization
            fast_weights = get_fast_weights(updates=updates, 
                                           initialization=fast_weights)

        # Get and return performance on query set
        test_preds = self.baselearner.forward_weights(test_x, fast_weights)
        test_loss = self.baselearner.criterion(test_preds, test_y)
        return test_loss, test_preds


    def train(self, train_x, train_y, test_x, test_y):
        """Train on a given task
        
        1. Evaluate the base-learner loss and gradients on the support set (train_x, train_y)
        using our initialization point.
        2. Make a single weight update based on this information.
        3. Evaluate the loss of the fast weights (initialization + proposed updates)
        4. Backpropagete the loss to update the initialization weights and the OSOMetaLearner
           responsible for proposing the base-learner weight updates
        
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

        self.baselearner.train()
        self.task_counter += 1
        self.dcounter += 1
        # Put all tensors on right device
        train_x, train_y, test_x, test_y = put_on_device(
                                            self.dev,
                                            [train_x, train_y,
                                            test_x, test_y])
        
        # Deploy TURTLE on task to compute test loss
        test_loss, _ = self._deploy(train_x, train_y, test_x, test_y)
       
        # Propagate loss backwards and update the initialization and
        # weights of the meta-learner
        test_loss.backward()

        # Clip gradients
        if not self.grad_clip is None:
            for p in self.initialization:
                p.grad = torch.clamp(p.grad, -self.grad_clip, +self.grad_clip)

        # self.init_grad_buffer = [self.init_grad_buffer[i] + self.initialization[i].grad for i in range(len(self.initialization))]
        # for i in range(len(self.meta_grad_buffers)):
        #     buffer = self.meta_grad_buffers[i]
        #     buffer = [buffer[j] + self.initialization[i].grad for i in range(len(self.initialization))]
        self.grad_buffer = [self.grad_buffer[i] + self.meta_params[i].grad for i in range(len(self.meta_params))]
        
        self.optimizer.zero_grad()
        for ml in self.metalearners:
            ml.zero_grad()

        if self.task_counter % self.meta_batch_size == 0:
            # Copy gradients from self.grad_buffer to gradient buffers in the initialization parameters
            for i, p in enumerate(self.meta_params):
                p.grad = self.grad_buffer[i]
            self.optimizer.step() 
            # Reset all stats for next update
            self.optimizer.zero_grad()
            for ml in self.metalearners:
                ml.zero_grad()
            self.grad_buffer = [torch.zeros(p.size(), device=self.dev) for p in self.meta_params]
            self.task_counter = 0

        if self.metasearch:
            if self.dcounter % 1000 == 0:
                print("Checkpoint -", self.dcounter)
            # pick new initialization with probability 0.1
            if np.random.rand() < 0.1:
                self.baselearner = self.baselearner_fn(**self.baselearner_args).to(self.dev)
                self.initialization, _= get_params(self.baselearner, self.dev)
                for p in self.initialization:
                    p.requires_grad = True
            
            # If switch limit has been reached, go over to initialization mode
            if self.dcounter == self.decouple:
                self.baselearner = self.baselearner_fn(**self.baselearner_args).to(self.dev)
                self.initialization, _= get_params(self.baselearner, self.dev)
                for p in self.initialization:
                    p.requires_grad = True
                
                self.meta_params = list(self.initialization)
                # Create gradient buffers that keep track of gradients across tasks
                self.grad_buffer = [torch.zeros(p.size()) for p in self.meta_params]
                self.optimizer = self.opt_fn(self.meta_params, lr=self.lr)

                self.metasearch = False
        
    def evaluate(self, train_x, train_y, test_x, test_y, **kwargs):
        """Evaluate on a given task
        
        Use the support set (train_x, train_y) to compute weight updates
        using the OSO. After the one-step update, evaluate and return the loss
        on the query set (test_x, test_y).
        
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
            
        Returns
        ----------
        test_loss
            Loss/accuracy of the base-learner on the query set after the proposed
            one-step update
        """
        
        self.baselearner.eval()
        # Put all tensors on right device
        train_x, train_y, test_x, test_y = put_on_device(
                                            self.dev,
                                            [train_x, train_y,
                                            test_x, test_y])

        # Deploy Turtle on task to compute test loss
        test_loss, preds = self._deploy(train_x, train_y, test_x, test_y)
            
        # If our objective is to minimize, return the loss,
        # else the accuracy
        if self.operator == min:
            return test_loss.item(), None
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
        state_dict meta-learner
            State dictionary of the meta-learner
        """
        
        # Copy initialization parameters
        init = [p.clone().detach() for p in self.initialization]
        return init, [ml.state_dict() for ml in self.metalearners]
    
    def load_state(self, state):
        """Load the given state into the meta-learner
        
        Parameters
        ----------
        state : [initialization, state_dict]
            Initialization parameters and state dictionary of the meta-learner
        """
        
        init, sd = state
        initialization = [p.clone() for p in init]
        for p in initialization:
            p.requires_grad = True
        self.initialization = initialization
        for i in range(len(sd)):
            self.metalearners[i].load_state_dict(sd[i])