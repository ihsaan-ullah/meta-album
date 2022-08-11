import pickle

class Algorithm:
    
    def __init__(self, baselearner_fn, baselearner_args, opt_fn, T, T_val, 
                 T_test, test_batch_size, lr, dev, **kwargs):
        self.baselearner_fn = baselearner_fn
        self.baselearner_args = baselearner_args
        self.opt_fn = opt_fn
        self.T = T
        self.T_val = T_val
        self.T_test = T_test
        self.test_batch_size = test_batch_size
        self.lr = lr
        self.dev = dev
        self.trainable = True
        self.init_score = -float("inf")
        
    def train(self, train_x, train_y, test_x, test_y):
        raise NotImplementedError()
        
    def evaluate(self, num_classes, train_x, train_y, test_x, test_y):
        raise NotImplementedError()
    
    def dump_state(self):
        raise NotImplementedError()
    
    def load_state(self, state):
        raise NotImplementedError()
        
    def store_file(self, filename):
        state = self.dump_state()
        with open(filename, "wb+") as f:
            pickle.dump(state, f)
    