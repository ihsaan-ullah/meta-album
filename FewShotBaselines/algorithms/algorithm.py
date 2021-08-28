import pickle

from .modules.utils import get_init_score_and_operator

class Algorithm:
    """Superclass for meta-learning algorithms
    
    Provides a framework for expected functionality of meta-learning
    algorithms
    
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
    init_score : float
        Initial score which can be used for validation purposes 
        (start from -infty in case of accuracy and take maximum score,
        and +infty in case of MSE and minimize)
    operator : function
        Whether to minimize the score of maximize the score. Function is
        either max or min. 
        
    Methods
    -------
    train(train_x, train_y, test_x, test_y)
        Perform a single training step on a given task
    
    evaluate(train_x, train_y, test_x, test_y)
        Evaluate the performance on the given task
    """
    
    def __init__(self, baselearner_fn, baselearner_args, opt_fn, T, T_val, T_test,
                 train_batch_size, test_batch_size, lr, dev, batching_eps, test_adam, **kwargs):
        """Initialization of the meta-learning algorithm
        
        Parameters
        ----------
        baselearner_fn : constructor function
            Constructor function for the baselearner
        baselearner_args : dict
            Keyword arguments for base-learner
        optim_fn : constructor function
            Constructor function for the optimizer to use
        T : int
            Number of update steps to parameters per task
        train_batch_size : int
            Indicating the size of minibatches that are sampled from meta-train tasks
        test_batch_size : int
            Size of batches to sample from meta-[val/test] tasks
        lr : float
            Learning rate for the optimizer
        dev : str
            Device to run model operations on
        """
        
        self.baselearner_fn = baselearner_fn
        self.baselearner_args = baselearner_args
        self.opt_fn = opt_fn
        self.T = T
        self.T_val = T_val
        self.T_test = T_test
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.lr = lr
        self.dev = dev
        self.trainable = True
        self.episodic = True
        self.batching_eps = batching_eps
        self.test_adam = test_adam
        # Initial score and which operator should be used for parameter selection
        self.init_score, self.operator = get_init_score_and_operator(
                                            baselearner_args["criterion"])
        
    def train(self, train_x, train_y, test_x, test_y):
        """Train on a given task
        
        Use the support set (train_x, train_y) and/or 
        the query set (test_x, test_y) to update the model.
        
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
        
        raise NotImplementedError()
        
    def evaluate(self, train_x, train_y, test_x, test_y):
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
        
        raise NotImplementedError()
    
    def dump_state(self):
        """Return the state of the model which can be loaded later
        
        Returns
        ----------
        state
            State of the model
        """
        
        raise NotImplementedError()
    
    def load_state(self, state):
        """Load the given state into the meta-learner
        
        Loads the given state of the model
        
        Parameters
        ----------
        state
            State of the model
        """
        
        raise NotImplementedError()
        
    def store_file(self, filename):
        """Stores the current state in the given file
        
        Parameters
        ----------
        filename : str
            Filename specifier (string)
        """
        
        state = self.dump_state()
        with open(filename, "wb+") as f:
            pickle.dump(state, f)
            
    def read_file(self, filename, **kwargs):
        """Reads the state from the given file and loads it into the model
        
        Parameters
        ----------
        filename : str
            Filename specifier (string)
        """
        
        with open(filename, "rb") as f:
            state = pickle.load(f)
        
        self.load_state(state, **kwargs)
        
        
            