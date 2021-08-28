import os
import numpy as np
import random
import pickle
import torch
from tqdm import tqdm

from PIL import Image

PATH = ["./data/cub/","./data/min/"]
SPLITS = ["train", "val", "test"]


def numpy_img_to_torch(batch):
    """Preprocess images
    
    Process numpy images:
    1. Normalize by dividing by 255.0
    2. Set datatype to be float32
    3. Make sure the input dimensions are (batch size, channels, height, width)
    4. Convert to torch.Tensor type
    
    Returns
    -------
    torch.Tensor
        Processed images ready to be fed as input to Conv2D
    """
    
    return torch.from_numpy((batch/255.0).astype('float32'))\
                .permute(0, 3, 1, 2)

class ImageLoader:
    """
    Data loader for images

    ...

    Attributes
    ----------
    N : int
        Number of classes per query/support set
    k : int
        Number of examples in all support sets
    k_test : int
        Number of examples in query sets of meta-validation and -test tasks
    img_size : int
        Reshape all images to this size
    class_map : dict
        Maps classes onto integers
    class_index : int
        Integer that will be assigned to a new class (after which it is incremented by 1)
    imdict : dict
        Stores images for episodic sampling. [mode]->[class]->list of examples
    images : dict
        Stores images for batch sampling. [mode]-> list of images
    labels
        Stores labels of images for batch sampling: [mode] -> list of labels

    Methods
    ----------
    _load_data()
        Prepare and load data into the loader object
    _sample_batch(self, size, mode)
        Sample a flat batch of data (no explicit task structure)
    _sample_episode(mode)
        Sample an episode consisting of a support and query set
    _draw_fn(return_props=False)
        Generates an actual sine function
    generator(mode, batch_size)
        Return a generator object that iterates over episodes
    """
    
    def __init__(self, N, k, k_test, img_size, path, cross=False, num_train_tasks=40000, 
                 num_val_tasks=600,seed=1337):  
        """
        initialize random seed used to generate sine wave data

        Parameters
        ----------
        N : int
            Number of classes to sample
        k : int
            Examples per class in support set
        k_test : int
            Number of examples per class in query sets
        img_size : int
            Reshape all images to this size
        path : str
            Root directory of images (which contains directories train/val/test)
        cross : boolean, optional
            Whether this data loader is used in cross-domain context. If so, merge 
            validation and test data into a single bin. (default=False)
        seed : int, optional
            Randoms seed to use
        """
        
        # Set seeds for reproducability. 
        np.random.seed(seed)
        random.seed(seed)
        
        self.N = N
        self.k = k
        self.k_test = k_test
        self.num_train_tasks = num_train_tasks
        self.num_val_tasks = num_val_tasks
        self.img_size = img_size # 2-tuple (width, height)
        self.class_index = 0
        # Store data for episodic sampling
        self.imdict = None
        # Store data for flat minibatches
        self.images = {split:[] for split in SPLITS}
        self.labels = {split:[] for split in SPLITS}
        self.tasks = {split:[] for split in SPLITS}
        
        # Pointers to current episode
        self.ptr = {"train":0, "val":0, "test":0} #eval:0

        # Load the image data. 
        self._load_images(path=path, dump=True, overwrite=False) 

    def _sample_batch(self, size, mode):
        """Sample a flat batch of data
        
        Samples a batch of data from all meta-training data
        
        Parameters
        ----------
        size : int
            Size of the batch to sample
        mode : str
            Current mode of operation, one of: train/val/test
            
        Returns
        ----------
        X
            Inputs of the batch
        Y
            Labels of the batch
        """
        
        num_images = len(self.images[mode])
        # use this instead of np randint to avoid seed interference
        indices = np.random.randint(0, num_images, size=size)
        # np.array([random.randint(0, num_images - 1) for _ in range(size)]) 
        X = self.images[mode][indices]
        Y = self.labels[mode][indices]
        return X, Y, None, None

    def _sample_episode(self, mode, **kwargs):  
        """Sample a single task
        
        Samples an episode consisting of a task for the given mode.
        1. Sample N classes uniformly at random
        2. For each of the N classes, sample k examples for the support set
           and k (if mode=train) or k_test (if mode = val/test) examples for 
           the query set. Append these in linear fashion.
        3. Shuffle support and query sets
        
        Parameters
        ----------
        mode : str
            Current mode of operation, one of: train/val/test
        **kwargs : dict
            Keyword arguments to ignore
        
        Returns
        ----------
        train_X, train_Y
            The support set with inputs train_X and labels train_Y
            Numpy arrays if self.use_torch=False, else torch.Tensors.
        test_X, test_Y
            The query set with inputs test_X and labels test_Y
            Numpy arrays if self.use_torch=False, else torch.Tensors.
        """
        
        idx = self.ptr[mode]
        self.ptr[mode] += 1

        support_indices, train_y, query_indices, test_y = self.tasks[mode][idx]
        train_x = self.images[mode][support_indices]
        test_x = self.images[mode][query_indices]

        return train_x, train_y, test_x, test_y
               
    def generator(self, episodic, batch_size, mode, reset_ptr, **kwargs):
        """Data generator
        
        Iterate over all tasks (if episodic), or for a fixed number of episodes (if episodic=False)
        and yield batches of data at every step.

        Parameters
        ----------
        episodic : boolean
            Whether to return a task (train_x, train_y, test_x, test_y) or a flat batch (x, y)
        batch_size : int
            Size of flat batch to draw (only applicable if episodic=False)
        mode : str
            "train"/"val"/"test": mode of operation
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

        # Number of data points a meta-learning system sees during meta-train time
        if mode == "train":
            num_iters = self.num_train_tasks
        else:
            num_iters = self.num_val_tasks
        
        if episodic:
            print(f"\n[*] Creating episodic generator for '{mode}' mode")
            gen_fn = self._sample_episode
        else:
            print(f"\n[*] Creating batch generator for '{mode}' mode")
            assert mode=="train", f"Tried to sample flat batch in '{mode}' mode"
            gen_fn = self._sample_batch
            # Print warning message if required
            if batch_size > self.k:                
                print(f"\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                print(f"WARNING: batch_size > k")
                print(f"@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

        for idx in range(num_iters): 
            yield gen_fn(mode=mode, size=batch_size)
    
    def total_classes(self, mode):
        """Get total number of classes for a given mode
        
        Parameters
        ----------
        mode : str
            Mode (train/val/test)
            
        Returns
        ----------
        int
            Number of classes in a certain mode
        """
        
        return len(np.unique(self.labels[mode]))
  
    
    def _load_images(self, path, dump=True, overwrite=False): 
        """Loads JPG images
        
        Assumption: path is structured as [train/val/test] -> [classes] -> [images]
        If so, then:
        1. Check if images from path have been loaded before. 
           If so, load them for speed!
        2. Else, we have to process all images ourselves
            a. Iterate over train/val/test folders
            b. Iterate over all classes in the folder
            c. Iterate over all images in the class folder
            d. Add the processed iamges 
            
        Parameters
        ----------
        path : str
            Root directory of images (with directories train/val/test)
        dump : boolean, optional
            Whether to store the processed images for next time
            (default=True)
        overwrite : boolean, optional
            Whether to overwrite an existing data.pkl with loaded images
            (default=False) 
        """
        
        # Name of the pickle file
        dat_file = f"N{self.N}k{self.k}test{self.k_test}-imgsize{self.img_size[0]}.pkl"
        
        # Check if earlier data dump exists. If so, return it
        if os.path.exists(os.path.join(path, dat_file)) and not overwrite:
            print(f"[*] Found previous data file. Loading now...")
            with open(os.path.join(path, dat_file),"rb") as f:
                self.images, self.labels, self.tasks = pickle.load(f)
                for split in ["train", "val", "test"]:
                    print(len(self.tasks[split]))
            return

        self.imdict = dict() # [split] -> [class] -> [indices of images with class]
        print(f"[*] Failed to find data file. Creating one now...")
        # No data.pkl exists, so we have to create one    
        # Traverse hierarchy of directories to fill imdict
        for split in SPLITS:
            image_ptr = 0
            print(f"     - Processing {split} images")
            self.imdict[split] = dict()
            # E.g. ./data/min/train/ or ./data/cub/val
            split_dir = os.path.join(path, split)
            for classdir in os.listdir(split_dir):
                # Create list to hold images with the given class <classdir> 
                self.imdict[split][classdir] = []
                self.class_index += 1
                    
                # Full path to the class directory
                full_class_dir = os.path.join(split_dir, classdir)
                
                # Iterate over all images of the given class and store them in the 
                # appropriate data structures
                for imagefile in os.listdir(full_class_dir):
                    # Ignore non-JPG images
                    if ".jpg" in imagefile:
                        # full path to image
                        image_location = os.path.join(full_class_dir, imagefile)
                        # Load the image using the PIL library
                        img = Image.open(image_location)
                        # Resize image to self.img_size
                        new_img = np.asarray(img.resize(self.img_size))
                        # neglect outliers which are not RGB 
                        if len(new_img.shape) > 2:
                            self.imdict[split][classdir].append(image_ptr)
                            self.images[split].append(new_img)
                            self.labels[split].append(self.class_index - 1)
                            image_ptr += 1
                # Convert image indices to numpy array for fast indexing
                self.imdict[split][classdir] = np.array(self.imdict[split][classdir], dtype=np.int32)
                
            # Convert images and labels to torch.Tensor for fast indexing
            self.images[split] = numpy_img_to_torch(np.array(self.images[split]))
            self.labels[split] = torch.from_numpy(np.array(self.labels[split]).astype("long"))
        
        # [split] -> [list of tasks]
        # every task is a tuple (support indices, query indices)
        self.tasks = {split:[] for split in SPLITS}

        split_to_count = {
            "train": self.num_train_tasks,
            "val": self.num_val_tasks,
            "test": self.num_val_tasks
        }

        print(split_to_count)

        # Examples in the support and query set for a single class
        total_size = self.k + self.k_test

        # Generate tasks
        print(f"[*] Generating tasks")
        for sid, split in enumerate(SPLITS):
            num_tasks = split_to_count[split]
            classes = list(self.imdict[split].keys())

            # Generate <num_tasks> tasks
            for i in range(num_tasks):
                N_classes = random.sample(classes, self.N)
                support_indices = np.array([], dtype=np.int32)
                query_indices = np.array([], dtype=np.int32)
                support_labels = []
                query_labels = []

                # For every of the N classes, sample k+k_test images
                for cid, classname in enumerate(N_classes):
                    pool = self.imdict[split][classname] # pool of image indices to pick from
                    indices = np.random.choice(pool, size=total_size)
                    
                    support_indices = np.concatenate([support_indices, indices[:self.k]])
                    query_indices = np.concatenate([query_indices, indices[self.k:]])

                    support_labels = support_labels + [cid for _ in range(self.k)]
                    query_labels = query_labels + [cid for _ in range(self.k_test)]

                # Randomly permute the support and query sets
                perm = np.random.permutation(len(support_indices))
                support_indices = support_indices[perm]
                support_labels = torch.Tensor(np.array(support_labels)[perm]).long()

                perm = np.random.permutation(len(query_indices))
                query_indices = query_indices[perm]
                query_labels = torch.Tensor(np.array(query_labels)[perm]).long()

                self.tasks[split].append(tuple([support_indices, support_labels, 
                                                query_indices, query_labels]))

        store_obj = tuple([self.images, self.labels, self.tasks])

        # Dump pickle file called data.pkl to avoid doing this entire 
        # time-consuming process again
        if dump:
            with open(os.path.join(path, dat_file),"wb+") as f:
                pickle.dump(store_obj, f, protocol=4)
