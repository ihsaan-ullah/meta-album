from .algorithm import Algorithm
from .modules.utils import put_on_device, deploy_on_task

   
class TrainFromScratch(Algorithm):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.trainable = False
        self.baselearner = self.baselearner_fn(**self.baselearner_args).to(
            self.dev)
        self.model = self.baselearner_fn(**self.baselearner_args).to(self.dev)
        
    def evaluate(self, num_classes, train_x, train_y, test_x, test_y, 
                 **kwargs):
        self.model.load_params(self.baselearner.state_dict())
        if num_classes is not None:
            self.model.modify_out_layer(num_classes)
        
        optimizer = self.opt_fn(self.model.parameters(), lr=self.lr)
        
        train_x, train_y, test_x, test_y = put_on_device(self.dev, [train_x, 
            train_y, test_x, test_y])
        test_score, loss_history, probs, preds = deploy_on_task(
            model=self.model, optimizer=optimizer, train_x=train_x, 
            train_y=train_y, test_x=test_x, test_y=test_y, T=self.T, 
            test_batch_size=self.test_batch_size)

        return test_score, loss_history, probs, preds
    
    def dump_state(self):
        return self.baselearner.state_dict()
    
    def load_state(self, state):
        self.baselearner.load_state_dict(state)
        