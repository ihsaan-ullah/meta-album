import os
import random
import numpy as np

import torch

from typing import List, Any

def set_random_seeds(random_seed: int) -> None:
    if random_seed is not None:
        torch.backends.cudnn.deterministic = False
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)


def lprint(text: str, 
           logs_path: str) -> None:
    print(text)
    with open(logs_path, "a") as f:
        f.write(text + "\n")


def get_device(logs_path: str) -> torch.device:
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        lprint(f"Using GPU: {torch.cuda.get_device_name(device)}", logs_path)
    else:
        device = torch.device("cpu")
        lprint("Using CPU", logs_path)
    return device


def get_torch_gpu_environment() -> List[str]:
    env_info = list()
    env_info.append(f"PyTorch version: {torch.__version__}")

    if torch.cuda.is_available():
        env_info.append(f"Cuda version: {torch.version.cuda}")
        env_info.append(f"cuDNN version: {torch.backends.cudnn.version()}")
        env_info.append("Number of available GPUs: "
            + f"{torch.cuda.device_count()}")
        env_info.append("Current GPU name: " +
            f"{torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        env_info.append("Number of available GPUs: 0")
    
    return env_info


def create_results_dir(res_dir: str,
                       logs_path: str) -> None:
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
        lprint(f"[+] Results directory created: {res_dir}", logs_path)
    else:
        lprint(f"[!] Results directory already exists: {res_dir}", logs_path)
        
        
def create_dir(dirname: str) -> None:
    if not os.path.exists(dirname):
        try:
            os.mkdir(dirname)
        except FileExistsError:
            pass


def count_trainable_parameters(model: Any) -> int:
    return sum([x.numel() for x in model.parameters() if x.requires_grad])
