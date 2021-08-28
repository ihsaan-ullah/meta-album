import os
import csv
import numpy as np
import torch
import pickle
import random

DIRNAME = "./data/sine/"
FILES = ["train.csv", "val.csv", "test.csv"]
SIZES = [70000, 1000, 2000] # Number of functions per split (train/val/test)
TOTAL = sum(SIZES)

class SineLoader:
    """
    Data loader for sine wave regression

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

        # Pointers to current episode
        self.ptr = {"train":0, "val":0, "test":0}

        self.k = k
        self.k_test = k_test
        # Store the actual functions in tuples (amplitude, phase)
        self.functions = {"train":[], "val":[], "test":[]}
        # Store function data [train/val/test] -> [index] -> train_x, train_y, test_x, test_y
        self.episodic_fn_data = {"train":[], "val":[], "test":[]}
        # Contains flat data from all the support and test sets together
        self.flat_fn_data = {"train":{"x":[], "y":[]}, "val":{"x":[], "y":[]}, "test":{"x":[], "y":[]}}
        # Load the data!
        self._load_data(total=TOTAL, sizes=SIZES)


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
        
        amplitude = np.random.uniform(0.1, 5.0)
        phase = np.random.uniform(0, np.pi)
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
    
    def _generate_data(self, k, k_test, fn, tensor=True):
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
        y = fn(x).reshape(-1, 1).astype('float32')
        train_x, train_y, test_x, test_y = x[:k], y[:k], x[k:], y[k:]
        if tensor:
            return torch.from_numpy(train_x), torch.from_numpy(train_y),\
                   torch.from_numpy(test_x), torch.from_numpy(test_y)
        return train_x, train_y, test_x, test_y 
  
    def _load_data(self, total, sizes):
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
        
        # Does directory ./data/sine exist?
        if not os.path.exists(DIRNAME):
            print(f"[*] Creating directory {DIRNAME}")
            os.mkdir(DIRNAME)
        else:
            print(f'[*] Directory {DIRNAME} already exists')
        
        dirstr = "k" + str(self.k) + "test" + str(self.k_test) +"/"  #(e.g. 'k5test100' or 'k10test150')
        wrdir = os.path.join(DIRNAME, dirstr) #./data/sine/k10test100/ e.g.

        # Does e.g. ./data/sine/k10test100/ already exist?
        if not os.path.exists(wrdir):
            print(f"[*] Creating directory: {wrdir}")
            os.mkdir(wrdir)
        else:
            print(f"[*] Directory {wrdir} already exists")
            
        # Data file example : ./data/sine/k10test100/dat.pkl 
        data_file = wrdir + "dat.pkl"

        if not os.path.exists(data_file):
            # Iterate over sizes for train/val/test splits
            for sid, size in enumerate(SIZES):
                prefix = FILES[sid].split(".")[0] # train/val/test used to index functions dict

                # Generate <size> functions 
                for j in range(size):
                    # Draw function and store in self.functions
                    fn, amplitude, phase = self._draw_fn(return_props=True)
                    tpl = str(amplitude), str(phase)
                    self.functions[prefix].append(tpl)

                    # Generate function data
                    train_x, train_y, test_x, test_y = self._generate_data(
                                                                k=self.k, 
                                                                k_test=self.k_test, 
                                                                fn=fn, 
                                                                tensor=True)

                    # Add episodic data
                    episode = train_x, train_y , test_x, test_y
                    self.episodic_fn_data[prefix].append(episode)

                    # Add flat batch data
                    self.flat_fn_data[prefix]["x"].append(train_x)
                    self.flat_fn_data[prefix]["x"].append(test_x)
                    self.flat_fn_data[prefix]["y"].append(train_y)
                    self.flat_fn_data[prefix]["y"].append(test_y)
                
                # Concatenate all tensors in flat_fn_data
                self.flat_fn_data[prefix]["x"] = torch.cat(self.flat_fn_data[prefix]["x"])
                self.flat_fn_data[prefix]["y"] = torch.cat(self.flat_fn_data[prefix]["y"])

            # Package all data and store in dat.pkl
            all_data = (self.functions, self.episodic_fn_data, self.flat_fn_data)
            with open(data_file, "wb+") as f:
                pickle.dump(all_data, f)
        else:
            print(f"[*] Opening {data_file} to read existing data")
            # Read data from pickle file and load into attributes
            with open(data_file, "rb") as f:
                all_data = pickle.load(f)
            self.functions, self.episodic_fn_data, self.flat_fn_data = all_data

    def _sample_batch(self, mode, size=None):
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

        indices = [random.randint(0, len(self.flat_fn_data[mode]["x"])-1) for _ in range(size)]
        indices = np.array(indices)
        x = self.flat_fn_data[mode]["x"][indices]
        y = self.flat_fn_data[mode]["y"][indices]
        return x, y, None, None

    def _sample_episode(self, mode, **kwargs):
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

        idx = self.ptr[mode]
        train_x, train_y, test_x, test_y = self.episodic_fn_data[mode][idx]
        self.ptr[mode] += 1
        return train_x, train_y, test_x, test_y

    def generator(self, episodic, batch_size, mode, reset_ptr=False, **kwargs):
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

        if reset_ptr:
            self.ptr[mode] = 0
        iters = len(self.episodic_fn_data[mode])
        # If episodic set number of iterations to the number of tasks
        if episodic:
            print(f"\n[*] Creating episodic generator for '{mode}' mode")
            gen_fn = self._sample_episode
        else:
            print(f"\n[*] Creating batch generator for '{mode}' mode")
            # Make sure mode is set to train, else do not allow the sampling of flat batches
            assert mode=="train", f"Tried to sample flat batch in '{mode}' mode"
            
            if batch_size > self.k:                
                # Print warning message
                print(f"\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                print(f"WARNING: batch_size > k")
                print(f"@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                
            gen_fn = self._sample_batch     
        
        print(f"[*] Generator set to perform {iters} iterations")
        for idx in range(iters):
            yield gen_fn(mode=mode, size=batch_size)
