import torch

# Train from scratch
TFS_CONF = {
    "opt_fn": torch.optim.Adam,
    "T": 100,
    "train_batch_size": 16,
    "test_batch_size": 4,
    "lr": 0.001,
}

# Fine Tuning
FT_CONF = {
    "opt_fn": torch.optim.Adam,
    "T": 100,
    "train_batch_size": 16,
    "test_batch_size": 4,
    "lr": 0.001,
}

# Model-agnostic meta-learning
MAML_CONF = {
    "opt_fn": torch.optim.Adam,
    "T": 5, 
    "lr": 0.001,
    "base_lr": 0.01,
    "meta_batch_size": 2
}

# Prototypical networks
PROTO_CONF = {
    "opt_fn": torch.optim.Adam,
    "lr": 0.001,
    "meta_batch_size":1,
    "T": 1
}

# Matching networks
MATCHING_CONF = {
    "opt_fn": torch.optim.Adam,
    "lr": 0.001,
    "meta_batch_size":1,
    "T": 1
}
