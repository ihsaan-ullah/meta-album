import torch
import numpy as np

from .algorithm import Algorithm
from .modules.utils import put_on_device, set_weights, get_loss_and_grads,\
                           empty_context, accuracy
from .modules.similarity import gram_linear, cka


class Reptile(Algorithm):
    """Reptile
    
    Meta-learning algorithm that attempts to obtain a good common 
    initialization point (base-learner parameters) across tasks.
    From this initialization point, we want to be able to make quick
    task-specific updates to achieve good performance from just few
    data points.
    Our implementation performs a single step of gradient descent
    
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
    
    def __init__(self, base_lr, no_annealing, grad_clip=None, meta_batch_size=1, **kwargs):
        """Initialization of reptile
        
        Parameters
        ----------
        T_test : int
            Number of updates to make at test time
        base_lr : float
            Learning rate for the base-learner 
        grad_clip : float
            Threshold for gradient value clipping
        meta_batch_size : int
            Number of tasks to compute outer-update
        **kwargs : dict
            Keyword arguments that are ignored
        """
        
        super().__init__(**kwargs)
        self.base_lr = base_lr
        self.grad_clip = grad_clip        
        self.meta_batch_size = meta_batch_size
        self.annealing = not no_annealing
        if self.annealing:
            self.meta_lr = 1
        else:
            self.meta_lr = self.lr
        # Increment after every train step on a single task, and update
        # init when task_counter % meta_batch_size == 0
        self.task_counter = 0 
        
        # Get random initialization point for baselearner
        self.baselearner = self.baselearner_fn(**self.baselearner_args).to(self.dev)
        self.initialization = [p.clone().detach().to(self.dev) for p in self.baselearner.parameters()]

        # Store gradients across tasks
        self.grad_buffer = [torch.zeros(p.size(), device=self.dev) for p in self.initialization]

        # Enable gradient tracking for the initialization parameters
        for p in self.initialization:
            p.requires_grad = True
                
        # Initialize the meta-optimizer, with Beta1 = 0
        self.optimizer = self.opt_fn(self.initialization, lr=self.base_lr, betas=(0, 0.999))
        self.opt_dict = self.optimizer.state_dict()

        # All keys in the base learner model that have nothing to do with batch normalization
        self.keys = [k for k in self.baselearner.state_dict().keys()\
                     if not "running" in k and not "num" in k]
        
        self.bn_keys = [k for k in self.baselearner.state_dict().keys()\
                        if "running" in k or "num" in k]


    def _get_params(self):
        return [p.clone().detach() for p in self.initialization]
          
    def _fast_weights(self, params, gradients):
        """Compute task-specific weights using the gradients
        
        Apply a single step of gradient descent using the provided gradients
        to compute task-specific, or equivalently, fast, weights.
        
        Parameters
        ----------
        params : list
            List of parameter tensors
        gradients : list
            List of torch.Tensor variables containing the gradients per layer
        """
        
        # Clip gradient values between (-10, +10)
        if not self.grad_clip is None:
            gradients = [torch.clamp(p, -self.grad_clip, +self.grad_clip) for p in gradients]
        
        fast_weights = [params[i] - self.base_lr * gradients[i]\
                        for i in range(len(gradients))]
        
        return fast_weights
    

    def _mini_batches(self, train_x, train_y, batch_size, num_batches, replacement):
        """
        Generate mini-batches from some data.
        Returns:
        An iterable of sequences of (input, label) pairs,
            where each sequence is a mini-batch.
        """
        if replacement:
            for _ in range(num_batches):
                print("train_x size[0]:", train_x.size()[0])
                ind = np.random.randint(0, train_x.size()[0], batch_size)
                yield train_x[ind], train_y[ind]
            return

        if batch_size == len(train_x):
            for i in range(num_batches):
                yield train_x, train_y
            return

        cur_batch = []
        batch_count = 0
        while True:
            ind = np.arange(len(train_x))
            np.random.shuffle(ind)
            lb = 0 # lowerbound
            #print("len:", len(train_x), "batch_num:", num_batches)
            while lb + batch_size <= len(train_x) and batch_count != num_batches:
                batch_count += 1
                lb += batch_size
                yield train_x[ind[lb-batch_size:lb]], train_y[ind[lb-batch_size:lb]] #We may exceed the upper bound by 1 but then we simply have a batch of size batch_size - 1
                
            if batch_count == num_batches:
                return

    def _deploy(self, train_x, train_y, test_x, test_y, train_mode, T, compute_cka=False):
        """Run reptile on a single task to get the loss on the query set
        
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
        train_mode : boolean
            Whether we are in training mode or test mode

        Returns
        ----------
        test_loss
            Loss of the base-learner on the query set after the proposed
            one-step update
        """
        
        orig_init = [p.clone().detach() for p in self.initialization]
        loss_history = None
        if not train_mode: loss_history = [] 
        if train_mode:
            # Join together the train and query sets in train mode and randomly sample
            # batches from that
            data_x, data_y = torch.cat([train_x, test_x]), torch.cat([train_y, test_y])
            optimizer = self.opt_fn(self.initialization, lr=self.base_lr, betas=(0, 0.999))
            optimizer.load_state_dict(self.opt_dict)
            batch_size = self.train_batch_size
        else:                
            optimizer = torch.optim.Adam(self.initialization, lr=self.base_lr, betas=(0,0.999))
            data_x, data_y = train_x, train_y
            batch_size = len(train_x) if len(train_x) <= 5 else 15

        batches = self._mini_batches(data_x, data_y, batch_size=batch_size, num_batches=T, replacement=False)
        for batch_x, batch_y in batches:
            loss, _ = get_loss_and_grads(self.baselearner, batch_x, batch_y, 
                                          weights=self.initialization, 
                                          create_graph=False,
                                          retain_graph=False,
                                          flat=False,
                                          rt_only_loss=True)
            if not train_mode: loss_history.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
        if compute_cka:
            fast_weights = [p.clone().detach() for p in self.initialization]

        if not train_mode:
            # Get and return performance on query set
            test_preds = self.baselearner.forward_weights(test_x, self.initialization)
            test_loss = self.baselearner.criterion(test_preds, test_y)
            for i in range(len(self.initialization)):
                self.initialization[i] = orig_init[i].clone().detach()
                self.initialization[i].requires_grad = True
            if not train_mode: loss_history.append(test_loss.item())
        else:
            self.opt_dict = optimizer.state_dict()
            for i in range(len(self.grad_buffer)):
                self.grad_buffer[i] =  self.grad_buffer[i] + (self.initialization[i].clone().detach() - orig_init[i])
                # Reset original initialization
                self.initialization[i] = orig_init[i].clone().detach()
                self.initialization[i].requires_grad = True
            test_preds, test_loss = None, None

        if compute_cka:
            return fast_weights
        return test_loss, test_preds, loss_history
    
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
        self._deploy(train_x, train_y, test_x, test_y, True, self.T)

        # Clip gradients
        if not self.grad_clip is None:
            for p in self.initialization:
                p.grad = torch.clamp(p.grad, -self.grad_clip, +self.grad_clip)

        if self.task_counter % self.meta_batch_size == 0: 
            self.task_counter = 0
            # Copy gradients from self.grad_buffer to gradient buffers in the initialization parameters
            for i, p in enumerate(self.initialization):
                self.initialization[i] = (self.initialization[i] + self.meta_lr * self.grad_buffer[i]/self.meta_batch_size).clone().detach()#.to(self.dev)
                self.initialization[i].requires_grad = True
                self.grad_buffer[i] = torch.zeros(p.size(), device=self.dev)
            if self.annealing:
                self.meta_lr -= 1/100000                  

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
        if val:
            T = self.T_val
        else:
            T = self.T_test


        if compute_cka:
            _, initial_features = self.baselearner.forward_weights_get_features(torch.cat((train_x, test_x)), weights=self.initialization)
            fast_weights = self._deploy(train_x, train_y, test_x, test_y, False, T, compute_cka=True)
            _, final_features = self.baselearner.forward_weights_get_features(torch.cat((train_x, test_x)), weights=fast_weights)
            ckas = []
            dists = []
            for features_x, features_y in zip(initial_features, final_features):
                ckas.append( cka(gram_linear(features_x), gram_linear(features_y), debiased=True) )
                dists.append( np.mean(np.sqrt(np.sum((features_y - features_x)**2, axis=1))) )
            # Compute the test loss after a single gradient update on the support set
            test_loss, preds, loss_history = self._deploy(train_x, train_y, test_x, test_y, False, T)
            preds = torch.argmax(preds, dim=1)
            return accuracy(preds, test_y), ckas, dists
        
        # Compute the test loss after a single gradient update on the support set
        test_loss, preds, loss_history = self._deploy(train_x, train_y, test_x, test_y, False, T)

        if self.operator == min:
            return test_loss.item(), loss_history
        else:
            # Turn one-hot predictions into class preds
            preds = torch.argmax(preds, dim=1)
            return accuracy(preds, test_y), loss_history
    
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
        
        self.initialization = [p.clone().detach().to(self.dev) for p in state]
        for p in self.initialization:
            p.requires_grad = True
        
    def to(self, device):
        self.baselearner = self.baselearner.to(device)
        self.initialization = [p.clone().detach().to(device) for p in self.initialization]
        for p in self.initialization:
            p.requires_grad = True