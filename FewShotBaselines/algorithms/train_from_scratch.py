from .algorithm import Algorithm
from .modules.utils import eval_model, get_batch, new_weights,\
                           put_on_device, update, deploy_on_task,\
                           get_init_score_and_operator
   
class TrainFromScratch(Algorithm):
    """
    Baseline that trains a base-learner network from scratch on every task
    it is presented with. No transfer of knowledge whatsoever.
    
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
    cpe : int
        Checkpoints per episode (# times we recompute new best weights)
    dev : str
        Device identifier
    trainable : boolean
        Whether it can train on meta-trrain tasks
    
    Methods
    -------
    evaluate(train_x, train_y, test_x, test_y)
        Evaluate the performance on the given task
        
    dump_state()
        Dumps the model state
        
    load_state(state)
        Loads the given model state
    """
    
    def __init__(self, cpe, **kwargs):
        """
        Call parent constructor function to inherit and set attributes
        Overwrite trainable attribute with FALSE, since this baseline is 
        can not be trained on meta-train tasks (it starts from scratch on all tasks)
        
        Parameters
        ----------
        cpe : int
            Number of times the best weights should be reconsidered in an episode
        """
        
        super().__init__(**kwargs)
        self.cpe = cpe
        self.trainable = False
        self.episodic = False
        # Initialize the base learner parameters
        self.baselearner = self.baselearner_fn(**self.baselearner_args).to(self.dev)
        # Copy of the model that will be used for training
        self.model = self.baselearner_fn(**self.baselearner_args).to(self.dev)
        
    def evaluate(self, train_x, train_y, test_x, test_y, **kwargs):
        """Evaluate the model on a given task
        
        Trains a base-learner from scratch for T epochs on randomly sampled 
        minibatches from the support set (train split) of size self.test_batch_size.
        Keeps track of best weights so far as evaluated on the entire train set, 
        and loads them to be evaluated on the query set (test_x, test_y).
        
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
        
        Returns
        ----------    
        test_score
            Floating point score on the query set (accuracy or MSE loss depending 
            on base-learner)
        """
        
        self.model.load_state_dict(self.baselearner.state_dict())
        optimizer = self.opt_fn(self.model.parameters(), lr=self.lr)
        
        # Put on the right device
        train_x, train_y, test_x, test_y = put_on_device(
                                            self.dev, 
                                            [train_x, train_y, 
                                             test_x, test_y])
        # Train on support set and get loss on query set
        test_score, loss_history = deploy_on_task(
                        model=self.model, 
                        optimizer=optimizer,
                        train_x=train_x, 
                        train_y=train_y, 
                        test_x=test_x, 
                        test_y=test_y, 
                        T=self.T, 
                        test_batch_size=self.test_batch_size,
                        cpe=self.cpe,
                        init_score=self.init_score,
                        operator=self.operator
        )

        return test_score, loss_history
    
    def dump_state(self):
        """Return the initial parameters of the base-learner
        
        Returns
        ----------
        state
            State of the model
        """
        
        return self.baselearner.state_dict()
    
    def load_state(self, state):
        """Load the given state into the base-learner
        
        Parameters
        ----------
        state
            State of the model
        """
        
        self.baselearner.load_state_dict(state)
        