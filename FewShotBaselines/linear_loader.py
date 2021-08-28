import os
import csv
import numpy as np
import torch
import pickle
import random

class LinearLoader:
    """
    Data loader for linear function regression

    ...

    Attributes
    -------
    ptr : dict
        Mapping of operation mode to episode index -> [train/val/test]->[episode index]
    k : int
        Number of examples in all support sets
    k_test : int
        Number of examples in query sets
    functions : dict
        Dictionary of functions -> [train/val/test] -> list of (amplitude,phase) pairs
    episodic_fn_data : dict
        Episode container [train/val/test]-> list of episodes (train_x, train_y, test_x, test_y)
    flat_fn_data : dict
        Container used for sampling flat batches (without task structure)
        [train/val/test]->[x/y]->all inputs/labels of that mode

    Methods
    -------
    _load_data()
        Prepare and load data into the loader object
    _sample_batch(self, size, mode)
        Sample a flat batch of data (no explicit task structure)
    _sample_episode(mode)
        Sample an episode consisting of a support and query set
    _draw_props()
        Generates a random amplitude and phase
    _draw_fn(return_props=False)
        Generates an actual sine function
    _get_fn(amplitude, phase)
        Returns a sine function with the given amplitude and phase
        -- Not used at the moment
    generator(mode, batch_size)
        Return a generator object that iterates over episodes
    """
    
    def __init__(self, k, k_test, seed=1337, **kwargs):
        """
        initialize random seed used to generate sine wave data

        Parameters
        -------
        k : int
            Sizes of support sets (and size of query set during meta-training time)
        k_test : int
            Sizes of query sets
        seed : int, optional
            Randoms seed to use
        **kwargs : dict, optional
            Trash can for optional arguments that are ignored (but has to stay for function call uniformity)
        """

        random.seed(seed)
        np.random.seed(seed)

        self.k = k
        self.k_test = k_test
        self.idx = 0
        self.pow = 2
        # Load the data!
        self._load_data()


    def _draw_props(self):
        """Generate random amplitude and phase

        Select amplitude and phase uniformly at random.
        Interval for amplitude : [0.1, 5.0]
        Interval for phase : [0, 3.14...(pi)]

        Returns
        ----------
        amplitude
            Amplitude of the sine function
        Phase
            Phase of the sine function
        """
        
        coefficient = np.random.uniform(-2, 2)
        bias = np.random.uniform(-2, 2)
        return amplitude, phase
    
    def _draw_fn(self, return_props=False):
        """Generate random sine function

        Randomly generate sine function fn that takes as input a real-valued x
        and returns y=fn(x) 
        The function has the form fn(x) = phase * np.sin(x + phase)

        Parameters
        ----------
        return_props : bool, optional
            Whether to return the amplitude and phase

        Returns
        ----------
        function
            The generated sine function
        amplitude (optional)
            Amplitude of the function
        phase (optional)
            Phase of the function
        """
        
        amplitude, phase = self._draw_props()
        
        def fn(x):
            return amplitude * np.sin(x + phase)
        if return_props:
            return fn, amplitude, phase
        
        return fn

    def _get_fn(self, amplitude, phase):
        """Construct sine function 

        Use the provided amplitude and phase to return the corresponding
        sine function

        Parameters
        ----------
        amplitude : float
            Amplitude of the function
        phase : float
            Phase of the function

        Returns
        ----------
        function
            The sine function with user-defined amplitude and phase
        """
        
        def fn(x):
            return amplitude * np.sin(x + phase)
        
        return fn
    
    def _generate_data(self, k, k_test, coeff, bias, tensor=True):
        """Generate input, output pairs for a given sine function

        Return input and output vectors x, y. Every y_i = fn(x_i) 

        Parameters
        ----------
        k : int
            Number of (x,y) pairs to generate
        k_test : int
            Number of examples in query set
        fn : function
            Sine function to use for data point generation
        tensor : bool, optional
            Whether to return x and y as torch.Tensor objects
            (default is np.array with dtype=float32)

        Returns
        ----------
        train_x
            Inputs of support set, randomly sampled from [-5,5]
        train_y
            Outputs of support set 
        test_x
            Inputs of query set drawn at random from [-5,5]
        test_y
            Outputs of query set
        """

        x = np.random.uniform(-5.0, 5.0, k+k_test).reshape(-1, 1).astype('float32')
        y = (coeff *x**self.pow + bias).reshape(-1, 1).astype('float32')
        train_x, train_y, test_x, test_y = x[:k], y[:k], x[k:], y[k:]
        if tensor:
            return torch.from_numpy(train_x), torch.from_numpy(train_y),\
                   torch.from_numpy(test_x), torch.from_numpy(test_y)
        return train_x, train_y, test_x, test_y 
  
    def _load_data(self):
        """Prepare data to be loaded
        
        Executes all functions related to the preparation of the data.
        This includes writing, reading, and creating object attributes to
        store the data.

        1. Check if old functions exist in folder 'k'+self.k+'test'+self.k_test
        2. Load old data if exists, else continue
        3. Randomly generate sine functions and data. Store in attributes
        4. Pickle all data in dat.pkl for future usage (for reproducibility and fairness of comparison)

        Parameters
        ----------
        total : int
            Number of sine functions
        sizes : list
            Sizes of train/val/test partitions : [train_size, val_size, test_size]
        """
        
        # List of coefficient-bias pairs
        # self.tasks = [(-1,1), (1,1), (1,-1), (-1,-1)] # zero-centered
        # self.tasks = [(2,2), (2,1), (1,2)] # traingular centered
        self.tasks = [(1,1), (1,2), (2,2), (2,1), (3,4), (4,4), (4,3), (3,3)]
        self.episodic_data = []
        batch_x = []
        batch_y = []

        count = 0
        while count < 100000:
            for coef, bias in self.tasks:
                train_x, train_y, test_x, test_y = self._generate_data(self.k, self.k_test, coef, bias)
                data_x, data_y = torch.cat((train_x, test_x)), torch.cat((train_y, test_y))
                self.episodic_data.append(tuple([train_x, train_y, test_x, test_y]))
                batch_x.append(data_x)
                batch_y.append(data_y)
            count += 1
        
        self.batch_x = torch.cat(batch_x)
        self.batch_y = torch.cat(batch_y)



    def _sample_batch(self, size=None):
        """Sample a flat batch of data
        
        Randomly sample @size data points from self.flat_fn_data[mode]

        Parameters
        ----------
        size : int
            Size of the data batch
        mode : str
            "train"/"val"/"test" indicating the mode of operation
        
        Returns
        ----------
        x
            torch.Tensor of size [size, 1] representing the inputs
        y 
            torch.Tensor of size [size,1] representing the outputs
        None
            empty query set inputs
        None
            empty query set outputs
        """

        indices = [random.randint(0, len(self.batch_x)) for _ in range(size)]
        indices = np.array(indices)
        x = self.batch_x[indices].reshape(-1,1)
        y = self.batch_y[indices].reshape(-1,1)
        #print(x.shape, y.shape)
        return x, y, None, None

    def _sample_episode(self, **kwargs):
        """Sample a single episode
        
        Look up and return the current episode for the given mode

        Parameters
        ----------
        mode : str
            "train"/"val"/"test": mode of operation
        **kwargs : dict
            Trashcan for additional args

        Returns 
        ----------
        train_x
            Inputs of support sets
        train_y
            Outputs of support set 
        test_x
            Inputs of query set
        test_y
            Outputs of query set
        """

        train_x, train_y, test_x, test_y = self.episodic_data[self.idx]
        self.idx += 1
        return train_x, train_y, test_x, test_y

    def generator(self, episodic, batch_size, **kwargs):
        """Data generator
        
        Iterate over all tasks (if episodic), or for a fixed number of episodes (if episodic=False)
        and yield batches of data at every step.

        Parameters
        ----------
        episodic : boolean
            Whether to return a task (train_x, train_y, test_x, test_y) or a flat batch (x, y)
        mode : str
            "train"/"val"/"test": mode of operation
        batch_size : int
            Size of flat batch to draw
        reset_ptr : boolean, optional
            Whether to reset the episode pointer for the given mode
        **kwargs : dict
            Other optional keyword arguments to keep flexibility with other data loaders which use other 
            args like N (number of classes)
        
        Returns 
        ----------
        generator
            Yields episodes = (train_x, train_y, test_x, test_y)
        """

        iters = len(self.episodic_data)
        # If episodic set number of iterations to the number of tasks
        if episodic:
            print(f"\n[*] Creating episodic generator")
            gen_fn = self._sample_episode
        else:
            print(f"\n[*] Creating batch generator")      
            if batch_size > self.k:                
                # Print warning message
                print(f"\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                print(f"WARNING: batch_size > k")
                print(f"@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                
            gen_fn = self._sample_batch     
        
        print(f"[*] Generator set to perform {iters} iterations")
        for idx in range(iters):
            yield gen_fn(size=batch_size)
