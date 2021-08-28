
class DataLoader:
    """
    Superclass for data loaders responsible for generating data.

    ...

    Methods
    -------
    _sample_batch(*args, **kwargs)
        Sample a flat batch of data (no explicit task structure)
    _sample_episode(*args, **kwargs)
        Sample an episode consisting of a support and query set
    generator(*args, **kwargs)
        Data generator that yields batches/tasks
    """

    def __init__(self, *args, **kwargs):
        """
        Initliaze the attributes of the DataLoader.
        """
        pass

    def _sample_batch(self, *args, **kwargs):
        """Sample flat data
        
        Sample a flat batch of data as often done in regular machine learning contexts

        Parameters
        ----------
        *args : iterable
            Positional arguments
        **kwargs : dict
            Keyword arguments

        Returns
        ----------
        data
            A flat batch of data (x, y)
        """

        raise NotImplementedError()

    def _sample_episode(self, *args, **kwargs):
        """Sample a task
        
        Samples data in the form of a task consisting of a support and query set.

        Parameters
        ----------
        *args : iterable
            Positional arguments
        **kwargs : dict
            Keyword arguments

        Returns
        ----------
        data
            A task (train_x, train_y, test_x, test_y) 
        """

        raise NotImplementedError()
    
    def generator(self, *args, **kwargs):
        """Generator object that yields batches or episodes 
        
        Iteratively yield a sampled batch or task

        Parameters
        ----------
        *args : iterable
            Positional arguments
        **kwargs : dict
            Keyword arguments
        
        Returns
        ----------
        data
            A task (train_x, train_y, test_x, test_y). When the generator samples
            batches of data, test_x, and test_y are None
        """

        raise NotImplementedError()
