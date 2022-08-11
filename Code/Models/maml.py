import torch
import torch.nn.functional as F

from .algorithm import Algorithm
from .modules.utils import put_on_device, get_loss_and_grads, accuracy


class MAML(Algorithm):
    
    def __init__(self, base_lr, second_order=False, grad_clip=None, 
                 meta_batch_size=1, **kwargs):
        super().__init__(**kwargs)
        self.base_lr = base_lr
        self.grad_clip = grad_clip        
        self.second_order = second_order
        self.meta_batch_size = meta_batch_size

        self.task_counter = 0 
        self.baselearner = self.baselearner_fn(**self.baselearner_args).to(
            self.dev)
        self.initialization = [p.clone().detach().to(self.dev) for p in 
            self.baselearner.parameters()]
        self.val_learner = self.baselearner_fn(**self.baselearner_args).to(
            self.dev)

        self.grad_buffer = [torch.zeros(p.size(), device=self.dev) for p in 
            self.initialization]

        for p in self.initialization:
            p.requires_grad = True
                
        self.optimizer = self.opt_fn(self.initialization, lr=self.lr)

    def _fast_weights(self, params, gradients):
        if self.grad_clip is not None:
            gradients = [torch.clamp(p, -self.grad_clip, +self.grad_clip) for p 
                in gradients]
        
        fast_weights = [params[i] - self.base_lr * gradients[i] for i in 
                range(len(gradients))]
        
        return fast_weights
    
    def _deploy(self, learner, train_x, train_y, test_x, test_y, T, fast_weights):
        loss_history = list()

        for _ in range(T):     
            xinp, yinp = train_x, train_y

            loss, grads = get_loss_and_grads(learner, xinp, yinp, 
                weights=fast_weights, create_graph=self.second_order,
                retain_graph=T > 1 or self.second_order, flat=False)

            loss_history.append(loss)
            fast_weights = self._fast_weights(params=fast_weights, gradients=grads)

        xinp, yinp = test_x, test_y

        out = learner.forward_weights(xinp, fast_weights)
        test_loss = learner.criterion(out, yinp)
        loss_history.append(test_loss.item())
        with torch.no_grad():
            probs = F.softmax(out, dim=1)
            preds = torch.argmax(probs, dim=1)
            acc = accuracy(preds, test_y)
            
        return acc, test_loss, loss_history, probs.cpu().numpy(), \
            preds.cpu().numpy()
    
    def train(self, train_x, train_y, test_x, test_y):
        self.baselearner.train()
        self.task_counter += 1

        train_x, train_y, test_x, test_y = put_on_device(self.dev, [train_x, 
            train_y, test_x, test_y])
        
        fast_weights = [p.clone() for p in self.initialization]   
        acc, test_loss, _, probs, preds = self._deploy(self.baselearner, 
            train_x, train_y, test_x, test_y, self.T, fast_weights)

        test_loss.backward()
            
        if self.grad_clip is not None:
            for p in self.initialization:
                p.grad = torch.clamp(p.grad, -self.grad_clip, +self.grad_clip)

        self.grad_buffer = [self.grad_buffer[i] + self.initialization[i].grad 
            for i in range(len(self.initialization))]
        self.optimizer.zero_grad()

        if self.task_counter % self.meta_batch_size == 0: 
            for i, p in enumerate(self.initialization):
                p.grad = self.grad_buffer[i]
            self.optimizer.step()
            
            self.grad_buffer = [torch.zeros(p.size(), device=self.dev) for p in 
                self.initialization]
            self.task_counter = 0
            self.optimizer.zero_grad()
            
        return acc, test_loss.item(), probs, preds

    def evaluate(self, num_classes, train_x, train_y, test_x, test_y, 
                 val=True):
        if num_classes is None:
            self.baselearner.eval()
            fast_weights = [p.clone() for p in self.initialization]   
            learner = self.baselearner
        else:
            self.val_learner.load_params(self.baselearner.state_dict())
            self.val_learner.eval()
            self.val_learner.modify_out_layer(num_classes)
            
            fast_weights = [p.clone() for p in self.initialization[:-2]] 
            initialization = [p.clone().detach().to(self.dev) for p in 
                self.val_learner.parameters()]
            fast_weights.extend(initialization[-2:])
            for p in fast_weights[-2:]:
                p.requires_grad = True
            learner = self.val_learner
        
        train_x, train_y, test_x, test_y = put_on_device(self.dev,
            [train_x, train_y, test_x, test_y])
        if val:
            T = self.T_val
        else:
            T = self.T_test
        
        acc, _, loss_history, probs, preds = self._deploy(learner, train_x, 
            train_y, test_x, test_y, T, fast_weights)

        return acc, loss_history, probs, preds
    
    def dump_state(self):
        return [p.clone().detach() for p in self.initialization]
    
    def load_state(self, state):
        self.initialization = [p.clone() for p in state]
        for p in self.initialization:
            p.requires_grad = True
