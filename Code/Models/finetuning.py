import torch

from .algorithm import Algorithm
from .modules.utils import update, put_on_device, deploy_on_task

OPT_TO_CONSTR = {
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD
}

class FineTuning(Algorithm):
    
    def __init__(self, freeze, test_lr=0.001, test_opt="adam", beta1=0.9, 
                 beta2=0.999, **kwargs):
        
        super().__init__(**kwargs)
        self.freeze = freeze
        self.test_opt = test_opt
        self.test_lr = test_lr
        self.beta1=beta1
        self.beta2=beta2
        self.baselearner = self.baselearner_fn(**self.baselearner_args).to(
            self.dev)
        self.baselearner.train()
        self.val_learner = self.baselearner_fn(**self.baselearner_args).to(
            self.dev)
        self.val_learner.load_state_dict(self.baselearner.state_dict())
        self.val_learner.train()

        self.optimizer = self.opt_fn(self.baselearner.parameters(), lr=self.lr, 
            betas=(self.beta1, self.beta2))

    def train(self, train_x, train_y, **kwargs):
        self.baselearner.train()
        train_x, train_y = put_on_device(self.dev, [train_x, train_y])
        acc, loss, probs, preds = update(self.baselearner, self.optimizer, 
            train_x, train_y)
        return acc, loss, probs, preds
        
    def evaluate(self, num_classes, train_x, train_y, test_x, test_y, 
                 val=True):
        self.val_learner.load_params(self.baselearner.state_dict())
        self.val_learner.eval()
        self.val_learner.freeze_layers(self.freeze, num_classes)
        
        opt_fn = OPT_TO_CONSTR[self.test_opt]
        val_optimizer = opt_fn(self.val_learner.parameters(), self.test_lr)
        
        # Put on the right device
        train_x, train_y, test_x, test_y = put_on_device(self.dev, [train_x, 
            train_y, test_x, test_y])
        
        if val:
            T = self.T_val
        else:
            T = self.T_test

        # Train on support set and get loss on query set
        test_acc, loss_history, probs, preds = deploy_on_task(
            model=self.val_learner, optimizer=val_optimizer, train_x=train_x, 
            train_y=train_y, test_x=test_x, test_y=test_y, T=T, 
            test_batch_size=self.test_batch_size)

        return test_acc, loss_history, probs, preds
        
    def dump_state(self):
        return {k : v.clone() for k, v in 
            self.baselearner.state_dict().items()}
    
    def load_state(self, state):
        self.baselearner.eval()
        self.baselearner.load_state_dict(state) 
