import contextlib
import math
import numpy as np
import torch
import torch.nn.functional as F


def accuracy(y_pred, y):
    return ((y_pred == y).float().sum()/len(y)).item()

    
def get_batch(train_x, train_y, batch_size):
    if batch_size is None:
        return train_x, train_y
    batch_indices = np.random.randint(0, train_x.size()[0], batch_size)
    x_batch, y_batch = train_x[batch_indices], train_y[batch_indices]
    return x_batch, y_batch


def update(model, optimizer, train_x, train_y):
    optimizer.zero_grad()
    out = model(train_x)
    loss = model.criterion(out, train_y)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        probs = F.softmax(out, dim=1)
        preds = torch.argmax(probs, dim=1)
        acc = accuracy(preds, train_y)
    return acc, loss.item(), probs.cpu().numpy(), preds.cpu().numpy()

    
def new_weights(model, best_weights, best_score, train_x, train_y):
    with torch.no_grad():
        eval_out = model(train_x)
        
        preds = torch.argmax(eval_out, dim=1)
        eval_score = accuracy(preds, train_y)
        
        tmp_best = max(eval_score, best_score)
        if tmp_best != best_score and not math.isnan(tmp_best):
            best_score = tmp_best
            best_weights = model.state_dict()
    return best_weights, best_score


def eval_model(model, x, y):
    with torch.no_grad():
        out = model(x)
        probs = F.softmax(out, dim=1)
        preds = torch.argmax(probs, dim=1)
        acc = accuracy(preds, y)
        loss = model.criterion(out, y).item()
    return acc, loss, probs.cpu().numpy(), preds.cpu().numpy()


def deploy_on_task(model, optimizer, train_x, train_y, test_x, test_y, T, 
                   test_batch_size):
    best_weights = model.state_dict() 
    
    loss_history = list()

    for t in range(T):        
        x_batch, y_batch = get_batch(train_x, train_y, test_batch_size)
        _, loss, _, _ = update(model, optimizer, x_batch, y_batch)
        loss_history.append(loss)

    if test_x is not None and test_y is not None:
        model.load_state_dict(best_weights)
        acc, loss, probs, preds = eval_model(model, test_x, test_y)
        loss_history.append(loss)
        return acc, loss_history, probs, preds


def process_cross_entropy(preds, targets, class_map, apply_softmax, dev, 
                          log=False, single_input=False):
    one_hot = torch.zeros((preds.size(0), 2*len(class_map.keys())), device=dev)
    if len(class_map.keys()) == 2:
        class_a, class_b = list(class_map.keys())
        one_hot[:, 0] = preds.view(-1)
        one_hot[:, 1] = 1 - preds.view(-1)
        if apply_softmax:
            one_hot[:,:2] = torch.softmax(one_hot[:,:2].clone(), dim=1)
        one_hot[targets == class_a, 2] = 1
        one_hot[targets == class_b, 3] = 1
        if log and not single_input:
            one_hot = torch.log(one_hot + 1e-5)

        outputs = one_hot[:,2].detach().float().view(-1,1)
        if single_input:
            if not log:
                one_hot = (one_hot[:,:2] * one_hot[:,2:]).sum(dim=1
                    ).unsqueeze(1)
            else:
                one_hot = torch.log((one_hot[:,:2] * one_hot[:,2:]).sum(dim=1
                    ).unsqueeze(1))


    else:
        outputs = torch.zeros(targets.size(), dtype=torch.long, device=dev)
        num_classes = len(class_map.keys())
        for c, column in class_map.items():
            column = class_map[c]
            one_hot[:, column] = preds[:, column]
            one_hot[targets == c, num_classes + column] = 1 
            outputs[targets == c] = column
        if apply_softmax:
            one_hot[:,:num_classes] = torch.softmax(one_hot[:,:num_classes
                ].clone(), dim=1)
        if log and not single_input:
            one_hot = torch.log(one_hot + 1e-5)
        if single_input:
            if not log:
                one_hot = (one_hot[:,:num_classes] * one_hot[:,num_classes:
                    ]).sum(dim=1).unsqueeze(1)
            else:
                one_hot = torch.log((one_hot[:,:num_classes] * 
                    one_hot[:,num_classes:]).sum(dim=1).unsqueeze(1))

    return one_hot, outputs


def get_loss_and_grads(model, train_x, train_y, flat=True, weights=None, 
                       item_loss=True, create_graph=False, retain_graph=False, 
                       rt_only_loss=False, meta_loss=False, class_map=None, 
                       loss_net=None, loss_params=None):
    model.zero_grad()
    if weights is None:
        weights = model.parameters()
        out = model(train_x)
    else:
        out = model.forward_weights(train_x, weights)
    
    if not meta_loss:
        loss = model.criterion(out, train_y)
    else:
        meta_inputs, targets = process_cross_entropy(out, train_y, 
            class_map=class_map, apply_softmax=True, dev=model.dev)
        loss = loss_net(meta_inputs, weights=loss_params)
    
    if rt_only_loss:
        return loss, None
    
    grads = torch.autograd.grad(loss, weights, create_graph=create_graph, 
        retain_graph=retain_graph)
    
    if flat:
        gradients = torch.cat([p.reshape(-1) for p in grads])
        loss = torch.zeros(gradients.size()).to(train_x.device) + loss.item()
    else:
        gradients = list(grads)
        if item_loss:
            loss = loss.item()
    return loss, gradients


def put_on_device(dev, tensors):
    for i in range(len(tensors)):
        if not tensors[i] is None:
            tensors[i] = tensors[i].to(dev)
    return tensors


@contextlib.contextmanager
def empty_context():
    yield None
    