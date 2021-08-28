import torch
import numpy as np

from .algorithm import Algorithm
from .modules.utils import put_on_device, set_weights, get_loss_and_grads,\
                           empty_context, accuracy, deploy_on_task
from .modules.similarity import gram_linear, cka


class MAML(Algorithm):
    """Model-Agnostic Meta-Learning
    
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
    
    def __init__(self, base_lr, second_order, grad_clip=None, meta_batch_size=1, special=False, **kwargs):
        """Initialization of model-agnostic meta-learner
        
        Parameters
        ----------
        T_test : int
            Number of updates to make at test time
        base_lr : float
            Learning rate for the base-learner 
        second_order : boolean
            Whether to use second-order gradient information
        grad_clip : float
            Threshold for gradient value clipping
        meta_batch_size : int
            Number of tasks to compute outer-update
        **kwargs : dict
            Keyword arguments that are ignored
        """
        
        super().__init__(**kwargs)
        self.sine = False
        self.base_lr = base_lr
        self.grad_clip = grad_clip        
        self.second_order = second_order
        self.meta_batch_size = meta_batch_size
        self.special = special        

        # Increment after every train step on a single task, and update
        # init when task_counter % meta_batch_size == 0
        self.task_counter = 0 
        
        # Maintain train loss history
        self.train_losses = []

        # Get random initialization point for baselearner
        self.baselearner = self.baselearner_fn(**self.baselearner_args).to(self.dev)
        self.initialization = [p.clone().detach().to(self.dev) for p in self.baselearner.parameters()]

        if self.special:
            self.val_learner = self.baselearner_fn(**self.baselearner_args).to(self.dev)

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
    
    def _deploy(self, train_x, train_y, test_x, test_y, train_mode, T, compute_cka=False):
        """Run DOSO on a single task to get the loss on the query set
        
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
        
        # if not train mode and self.special, we need to use val_params instead of self.init
        if self.special and not train_mode:
            learner = self.val_learner
            fast_weights = [p.clone() for p in self.val_params]
        else:
            fast_weights = [p.clone() for p in self.initialization]   
            learner = self.baselearner
        
        loss_history = None
        if not train_mode: loss_history = []
        if train_mode:
            # If batching episodic data, create random batches and train on them
            if self.batching_eps:
                #Create random permutation of rows in test set
                perm = torch.randperm(test_x.size()[0])
                data_x = test_x[perm]
                data_y = test_y[perm]

                # Compute batches
                batch_size = test_x.size()[0]//T
                batches_x = torch.split(data_x, batch_size)
                batches_y = torch.split(data_y, batch_size)

                batches_x = [torch.cat((train_x, x)) for x in batches_x]
                batches_y = [torch.cat((train_y, y)) for y in batches_y]

        for step in range(T):     
            if self.batching_eps and train_mode:
                xinp, yinp = batches_x[step], batches_y[step]
            else:
                xinp, yinp = train_x, train_y

            # if self.special and not train mode, use val_learner instead
            loss, grads = get_loss_and_grads(learner, xinp, yinp, 
                                        weights=fast_weights, 
                                        create_graph=self.second_order,
                                        retain_graph=T > 1 or self.second_order,
                                        flat=False)

            if not train_mode: loss_history.append(loss)

            fast_weights = self._fast_weights(params=fast_weights, gradients=grads)

        if compute_cka:
            return fast_weights

        if train_mode and T > 0:
            self.train_losses.append(loss)

        xinp, yinp = test_x, test_y

        # Get and return performance on query set
        test_preds = learner.forward_weights(xinp, fast_weights)
        test_loss = learner.criterion(test_preds, yinp)
        if not train_mode: loss_history.append(test_loss.item())
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
        test_loss, _,_ = self._deploy(train_x, train_y, test_x, test_y, True, self.T)

        # Propagate the test loss backwards to update the initialization point
        test_loss.backward()
            
        # Clip gradients
        if not self.grad_clip is None:
            for p in self.initialization:
                p.grad = torch.clamp(p.grad, -self.grad_clip, +self.grad_clip)

        self.grad_buffer = [self.grad_buffer[i] + self.initialization[i].grad for i in range(len(self.initialization))]
        self.optimizer.zero_grad()

        if self.task_counter % self.meta_batch_size == 0: 
            # Copy gradients from self.grad_buffer to gradient buffers in the initialization parameters
            for i, p in enumerate(self.initialization):
                p.grad = self.grad_buffer[i]
            self.optimizer.step()
            
            self.grad_buffer = [torch.zeros(p.size(), device=self.dev) for p in self.initialization]
            self.task_counter = 0
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
        loss_history = []
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
        
        if self.special:
            # copy initial weights except for bias and weight of final dense layer
            val_init = [p.clone().detach() for p in self.initialization[:-2]]
            self.val_learner.eval()
            # forces the network to get a new final layer consisting of eval_N classes
            self.val_learner.freeze_layers(freeze=False)
            newparams = [p.clone().detach() for p in self.val_learner.parameters()][-2:]

            self.val_params = val_init + newparams
            for p in self.val_params:
                p.requires_grad = True
            
            # Compute the test loss after a single gradient update on the support set
            test_loss, preds, loss_history = self._deploy(train_x, train_y, test_x, test_y, False, T)
            
        else:

            if self.test_adam:
                opt = self.opt_fn(self.baselearner.parameters(), self.lr)
                for p,q in zip(self.initialization, self.baselearner.parameters()):
                    q.data = p.data
                self.baselearner.train()
                test_acc, loss_history = deploy_on_task(
                                        model=self.baselearner, 
                                        optimizer=opt,
                                        train_x=train_x, 
                                        train_y=train_y, 
                                        test_x=test_x, 
                                        test_y=test_y, 
                                        T=T, 
                                        test_batch_size=self.test_batch_size,
                                        cpe=0.5,
                                        init_score=self.init_score,
                                        operator=self.operator        
                                    )
                return test_acc, loss_history 
            

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
                if self.sine:
                    return test_loss.item(), ckas, dists
                
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
        
        self.initialization = [p.clone() for p in state]
        for p in self.initialization:
            p.requires_grad = True
        
    def to(self, device):
        self.baselearner = self.baselearner.to(device)
        self.initialization = [p.to(device) for p in self.initialization]