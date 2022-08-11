import sys
import os

# in order to import modules from packages in the parent directory
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import time
import argparse
import glob
import datetime
import numpy as np
import pandas as pd

from torch.nn import CrossEntropyLoss
from torch.optim import SGD

import avalanche
from avalanche.training.strategies import Naive, Cumulative, AGEM, GEM, \
    Replay
from avalanche.evaluation.metrics import accuracy_metrics,\
    loss_metrics, ram_usage_metrics, cpu_usage_metrics,\
    disk_usage_metrics, gpu_usage_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin

from utils.utils import *
from utils.modified_avalanche_src import EWC
from Models.multihead_resnet18 import MultiHeadResnet18
from DataLoader.data_loader_continual_learning import ContinualLearningDataset

from typing import Any


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Continual learning experiment on Meta-Album")
    
    # Strategy config
    parser.add_argument("--strategy", type=str, default="Naive",
        help="continual learning strategy to use. Default. Naive.")
    parser.add_argument("--num_workers", type=int, default=4, 
        help="number of subprocesses to use for data loading. Default: 4.")
    parser.add_argument("--replay_mem_size", type=int, default=200,
        help="replay buffer size. Default: 200.")
    parser.add_argument("--agem_patterns_per_exp", type=int, default=64,
        help="number of patterns per experience in the memory. Default: 64.")
    parser.add_argument("--agem_sample_size", type=int, default=64,
        help="number of patterns in memory sample when computing reference " +
        "gradient. Default: 64.")
    parser.add_argument("--gem_patterns_per_exp", type=int, default=64,
        help="number of patterns per experience in the memory. Default: 64.")
    parser.add_argument("--gem_memory_strength", type=float, default=0.5,
        help="offset to add to the projection direction in order to favour " +
        "backward transfer. Default: 0.5.")
    parser.add_argument("--ewc_lambda", type=float, default=0.5,
        help="hyperparameter to weight the penalty inside the total loss. " +
        "Default: 0.5.")
    parser.add_argument("--ewc_mode", type=str, default="separate", 
        help="mode for the EWC strategy, it can be: separate, onlinesum, or " +
        "onlineweightedsum. Default: separate.")
    parser.add_argument("--ewc_decay_factor", type=float, default=None, 
        help="decay term of the importance matrix. Default: None.")
    parser.add_argument("--ewc_keep_importance_data", type=bool, default=False, 
        help="if True, keep in memory both parameter values and importances " +
        "for all previous task, for all modes. If False, keep only last " +
        "parameter values and importances. Default: False.")
    
    # Data loader config
    parser.add_argument("--train_test_ratio", type=float, default=0.7, 
        help="ratio between train and test images per dataset. Default 0.")
    parser.add_argument("--sequence_idx", type=int, default=0, 
        help="row of the latin square to generate the sequence. Default 0.")
    parser.add_argument("--random_seed", type=int, default=93,
        help="seed for random generators. Default: 93.")
    
    # Training config
    parser.add_argument("--epochs", type=int, default=20, 
        help="number of epochs to train the model. Default: 20.")
    parser.add_argument("--train_batch_size", type=int, default=64,
        help="number of images per batch to train the model. Default: 64.")
    parser.add_argument("--eval_batch_size", type=int, default=64,
        help="number of images per batch to evaluate the model. Default: 64.")
    
    # Optimizer config
    parser.add_argument("--lr", type=float, default=1e-2, 
        help="learning rate for the SGD optimizer. Default: 1e-2.")
    parser.add_argument("--momentum", type=float, default=0,
        help="momentum for the SGD optimizer. Default: 0.")
    parser.add_argument("--weight_decay", type=float, default=0,
        help="weight decay for the SGD optimizer. Default: 0.")
    
    args = parser.parse_args()
    args.img_size = 224
    
    return args


class ContinualLearningExperiment:
    
    # Initialization
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

        # Define paths
        self.curr_dir = os.path.dirname(__file__)
        self.main_dir = os.path.dirname(self.curr_dir)
        results_dir = "Results"
        experiment_results_dir = "continual_learning"
        run_id = f"run_{args.random_seed}_" + \
            f"{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        self.run_results_dir = os.path.join(self.main_dir, results_dir, 
            experiment_results_dir, run_id)
        logs_path = os.path.join(self.run_results_dir, "logs.txt")
        
        # Initialization step
        create_results_dir(self.run_results_dir, logs_path)
        set_random_seeds(args.random_seed)
        self.init_data_loader()
        self.device = get_device(logs_path)
        self.gpu_info = get_torch_gpu_environment()
        self.clprint = lambda text: lprint(text, logs_path)

        self.clprint("\n".join(self.gpu_info)) 

    def init_data_loader(self) -> None:
        data_dir = os.path.join(self.main_dir, "Data")
        dataset_list = list()
        for path in glob.glob(os.path.join(data_dir, "*", "labels.csv")):
            info = path.split(os.sep)
            dataset_list.append(f"{os.sep}".join(info[:-1]))
        dataset_list = sorted(dataset_list)
        
        self.dataset = ContinualLearningDataset(dataset_list, 
            self.args.img_size, self.args.train_test_ratio,
            self.args.random_seed)

    def get_model(self) -> Any: 
        self.model = MultiHeadResnet18().to(self.device)
        self.model_name = self.model.model_name

    def get_eval_plugin(self, scenario):
        eval_plugin = EvaluationPlugin(
            accuracy_metrics(experience=True, trained_experience=False),
            loss_metrics(experience=True),
            cpu_usage_metrics(experience=True),
            disk_usage_metrics(experience=True),
            gpu_usage_metrics(gpu_id=0, experience=True),
            ram_usage_metrics(experience=True),
            loggers=[InteractiveLogger()],
            benchmark=scenario
        )
        return eval_plugin

    def get_strategy(self, eval_plugin: Any) -> Any:
        kwargs = {"model": self.model,
                  "optimizer": SGD(self.model.parameters(),
                                   lr=self.args.lr,
                                   momentum=self.args.momentum,
                                   weight_decay=self.args.weight_decay),
                  "criterion": CrossEntropyLoss(),
                  "train_mb_size": self.args.train_batch_size,
                  "train_epochs": self.args.epochs,
                  "eval_mb_size": self.args.eval_batch_size,
                  "device": self.device,
                  "evaluator": eval_plugin,
                  "eval_every": -1}
        
        if self.args.strategy == "Naive":
            """ Naive finetuning.

            The simplest (and least effective) Continual Learning strategy. Naive 
            just incrementally fine tunes a single model without employing any 
            method to contrast the catastrophic forgetting of previous knowledge.
            This strategy does not use task identities.

            Naive is easy to set up and its results are commonly used to show the 
            worst performing baseline.
            """
            strategy = Naive(**kwargs)
        
        elif self.args.strategy == "Cumulative":
            """
            At each experience, train model with data from all previous experiences
            and current experience.
            """
            strategy = Cumulative(**kwargs)
        
        elif self.args.strategy == "Replay":
            """
            Experience replay strategy.
            mem_size: Replay buffer size.
            """
            strategy = Replay(mem_size=self.args.replay_mem_size, **kwargs)
        
        elif self.args.strategy == "AGEM":
            """ Averaged Gradient Episodic Memory.
            patterns_per_exp: Number of patterns per experience in the memory
            sample_size: Number of patterns in memory sample when computing
                reference gradient.
            """
            strategy = AGEM(patterns_per_exp=self.args.agem_patterns_per_exp,
                sample_size=self.args.agem_sample_size, **kwargs)
        
        elif self.args.strategy == "GEM":
            """ Gradient Episodic Memory.
            patterns_per_exp: Number of patterns per experience in the memory
            memory_strength: Offset to add to the projection direction in order to 
                favour backward transfer (gamma in original paper).
            """
            strategy = GEM(patterns_per_exp=self.args.gem_patterns_per_exp,
                memory_strength=self.args.gem_memory_strength, **kwargs)
        
        elif self.args.strategy == "EWC":
            """ Elastic weight consolidation
            It constrains the parameters to stay in a region of low error for 
            previous tasks by adding a regularization term to the loss function.
            ewc_lambda: Hyperparameter to weigh the penalty inside the total
                loss. The larger the lambda, the larger the regularization.
            mode: 'separate' to keep a separate penalty for each previous
                experience. 'onlinesum' to keep a single penalty summed over all
                previous tasks. 'onlineweightedsum' to keep a single penalty summed
                with a decay factor over all previous tasks.
            decay_factor: Used only if mode is 'onlineweightedsum'. It specify the 
                decay term of the importance matrix.
            keep_importance_data: If True, keep in memory both parameter values and
                importances for all previous task, for all modes. If False, keep 
                only last parameter values and importances. If mode is 'separate', 
                the value of 'keep_importance_data' is set to be True.
            """
            strategy = EWC(ewc_lambda=self.args.ewc_lambda, 
                mode=self.args.ewc_mode, 
                decay_factor=self.args.ewc_decay_factor,
                keep_importance_data=self.args.ewc_keep_importance_data, 
                **kwargs)
        
        else:
            raise Exception(f"Strategy '{self.args.strategy}' not valid.")
    
        return strategy

    # Model training and evaluation
    def update_test_acc_matrix(self,
                               score_matrix: np.ndarray, 
                               train_exp_idx: int, 
                               eval_exp_end: int, 
                               eval_last_metrics: Any) -> np.ndarray:
        for k, v in eval_last_metrics.items():
            k_split = k.split("/")
            if k_split[0] != "Top1_Acc_Exp":
                continue
            if k_split[1] != "eval_phase":
                continue
            if k_split[2] != "test_stream":
                continue
            if k_split[-1][:3] != "Exp":
                continue
            eval_exp_idx = int(k_split[-1][3:])
            if eval_exp_idx < eval_exp_end:
                score_matrix[train_exp_idx, eval_exp_idx] = v
        return score_matrix
    
    def compute_forgetting(self, 
                           score_matrix: np.ndarray, 
                           k: int) -> float:
        if k == 0:
            return 0
        sum_ = 0
        for i in range(k):
            sum_ += score_matrix[i, i] - score_matrix[k, i]
        return sum_ / k

    def compute_average_accuracy(self, 
                                 score_matrix: np.ndarray, 
                                 k: int) -> float:
        sum_ = 0
        for i in range(k + 1):
            sum_ += score_matrix[i, i]
        return sum_ / (k+1)

    def run(self) -> None:
        self.clprint("\n\n### ------------------------------------------ ###")
        self.clprint(f"### Strategy: {self.args.strategy}")
        self.clprint(f"### Sequence: {self.args.sequence_idx}")
        self.clprint(f"### Random Seed: {self.args.random_seed}")
        self.clprint("### ------------------------------------------ ###\n")
        
        self.get_model()
        scenario = self.dataset.get_scenario(self.args.sequence_idx) 
        eval_plugin = self.get_eval_plugin(scenario)
        strategy = self.get_strategy(eval_plugin)
        
        self.save_experimental_settings()
        
        score_matrix = -1 * np.ones((len(scenario.train_stream), 
            len(scenario.train_stream)), dtype=float)
        
        results = list()
        start_time = time.time()
        for train_exp_idx, experience in enumerate(scenario.train_stream):
            self.clprint("Start of experience: " +
                f"{experience.current_experience}, task label: " +
                f"{experience.task_label}, number of current classes: " +
                f"{len(experience.classes_in_this_experience)}")
            
            train_start = time.time()
            strategy.train(experience, num_workers=self.args.num_workers)
            elapsed_time_train = time.time() - train_start
            
            eval_exp_end = min(train_exp_idx + 1, len(scenario.test_stream))
            
            eval_start = time.time()
            eval_last_metrics = strategy.eval(
                scenario.test_stream[:eval_exp_end], 
                num_workers=self.args.num_workers)
            elapsed_time_eval = time.time() - eval_start 
            elapsed_time_eval /= len(scenario.test_stream[:eval_exp_end])
            
            self.clprint(f"Trained on task {experience.current_experience}, " +
                f"eval_last_metrics={eval_last_metrics}")
            
            score_matrix = self.update_test_acc_matrix(score_matrix, 
                train_exp_idx, eval_exp_end, eval_last_metrics)
            print(score_matrix)
            
            F_k = self.compute_forgetting(score_matrix, train_exp_idx)
            A_k = self.compute_average_accuracy(score_matrix, train_exp_idx)
            
            self.clprint(f"Task ID: {train_exp_idx}\tF_k: {F_k}\tA_k: {A_k}\t"+
                f"Elapsed training time: {elapsed_time_train}\tElapsed " + 
                f"training time: {elapsed_time_eval}")
            
            results.append([train_exp_idx, A_k, F_k, elapsed_time_train, 
                elapsed_time_eval])
            self.clprint("*" * 40)
            
        elapsed_time = time.time() - start_time
        
        # End experiment
        self.clprint(f"\nRun completed in: {elapsed_time//60}m " +
            f"{elapsed_time%60:.0f}s")
        self.save_experiment_results(results, score_matrix)
    
    # Results saving
    def save_experimental_settings(self) -> None:
        join_list = lambda info: "\n".join(info)
        
        gpu_settings = ["\n--------------- GPU settings ---------------"]
        gpu_settings.extend(self.gpu_info)
        
        model_settings = [
            "\n--------------- Model settings ---------------",
            f"Model: {self.model_name}",
            f"Trainable parameters: {count_trainable_parameters(self.model)}",
            f"Training strategy: {self.args.strategy}",
            f"Random seed: {self.args.random_seed}",
            "Loss function: Cross entropy",
            "Optimizer: SGD",
            f"Learning rate: {self.args.lr}",
            f"Momentum: {self.args.momentum}",
            f"Weight decay: {self.args.weight_decay}"
        ]
        
        training_settings = [
            "\n--------------- Training settings ---------------",
            f"Epochs: {self.args.epochs}",
            f"Train batch size: {self.args.train_batch_size}",
            f"Evaluation batch size: {self.args.eval_batch_size}",
            f"Train test ratio: {self.args.train_test_ratio}",
            f"Latin square row (sequence idx): {self.args.sequence_idx}",
            "Image size: 224"
        ]
        
        strategy_settings = [
            "\n--------------- Strategy settings ---------------",
            f"Num workers: {self.args.num_workers}",
            f"Replay memory size (Replay): {self.args.replay_mem_size}",
            "Patterns per experience (AGEM): " +
            f"{self.args.agem_patterns_per_exp}",
            f"Sample size (AGEM): {self.args.agem_sample_size}",
            f"Patterns per experience (GEM): {self.args.gem_patterns_per_exp}",
            f"Memory strength (GEM): {self.args.gem_memory_strength}",
            f"Lambda (EWC): {self.args.ewc_lambda}",
            f"Mode (EWC): {self.args.ewc_mode}",
            f"Decay factor (EWC): {self.args.ewc_decay_factor}",
            f"Keep importance data (EWC): {self.args.ewc_keep_importance_data}"
        ]
        
        all_settings = [
            "Description: This experiment is named 'Continual learning. In " +
            "this experiment the available datasets in Meta-Album are " +
            "randomly ordered. Then, each dataset is randomly split into a " +
            "training set and a test set. The continual learning agent " +
            "iteratively is trained on the dataset sequence (training set), " +
            "one at a time. Each time the training of one dataset finishes, " +
            "the agent is tested on the current dataset as well as all " +
            "previous datasets (test sets).",
            join_list(gpu_settings),
            join_list(model_settings),
            join_list(training_settings),
            join_list(strategy_settings)
        ]

        experimental_settings_path = os.path.join(self.run_results_dir, 
            "experimental_settings.txt")
        with open(experimental_settings_path, "w") as f:
            f.writelines(join_list(all_settings))
  
    def save_experiment_results(self, 
                                results: List[list], 
                                score_matrix: np.ndarray) -> None:
        # Save experiment results
        results_df = pd.DataFrame(results, columns=["Trained task position", 
            "A_k", "F_k", "Time to train 1 task", "Time to eval 1 task"])
        results_df.to_csv(os.path.join(self.run_results_dir, 
            "experiment_results.csv"), sep=",", encoding="utf-8")
        
        # Save predictions (score matrix in the context of continual learning)
        np.savetxt(os.path.join(self.run_results_dir, "predictions.csv"), 
            score_matrix, fmt="%1.3f", delimiter=",", newline="\n")


if __name__ == "__main__":
    # Check avalanche version
    supported_avalanche_version = "0.1.0"
    assert avalanche.__version__ == supported_avalanche_version, \
        f"Only avalanche {supported_avalanche_version} is supported."
    
    # Get args 
    args = parse_arguments()
    
    # Initiaize experiment object
    experiment = ContinualLearningExperiment(args)
    
    # Run experiment
    experiment.run()
