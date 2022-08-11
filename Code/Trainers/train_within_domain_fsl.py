#-----------------------
# Imports
#-----------------------

import sys
import os

# in order to import modules from packages in the parent directory
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import csv
import math
import numpy as np
import time
import random
import datetime
import pandas as pd
import scipy.stats as st
import pickle
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.utils import *
from DataLoader.data_loader_within_domain_fsl import CategoriesSampler, \
    FewshotDataset, extract_info, train_test_split_classes, process_labels
from Models.fsl_resnet import ResNet
from Models.train_from_scratch import TrainFromScratch
from Models.finetuning import FineTuning
from Models.protonet import PrototypicalNetwork
from Models.matchingnet import MatchingNetwork
from Models.maml import MAML

from utils.configs import TFS_CONF, FT_CONF, MAML_CONF, PROTO_CONF, \
    MATCHING_CONF


#-----------------------
# Arguments
#-----------------------
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument("--dataset", type=str, required=True, 
        help="which dataset to use.")
    parser.add_argument("--model", choices=["tfs", "finetuning", "maml", 
        "protonet", "matchingnet"], required=True, help="which model to use")
    
    # Optional arguments

    # Episodes config
    parser.add_argument("--N_train", type=int, default=None,
        help="number of ways for the support set of the training episodes. " + 
        "Default: None.")
    parser.add_argument("--k_train", type=int, default=None,
        help="number of shots for the support set of the training episodes. " +
        "Default: None.")
    parser.add_argument("--k_test", type=int, default=16, 
        help="number of images for the query set of the testing episodes. " +
        "Default: 16.")
    parser.add_argument("--train_batch_size", type=int, default=None,
        help="size of minibatches for training only applies for flat batch " +
        "models. Default: None.")
    
    parser.add_argument("--N", type=int, default=5,
        help="number of ways for the support set of the testing episodes, " +
        "if None, the episodes are any-way tasks. Default: 5.")
    parser.add_argument("--k", type=int, default=1,
        help="number of shots for the support set of the testing episodes, " +
        "if None, the episodes are any-shot tasks. Default: 1.")
    parser.add_argument("--test_batch_size", type=int, default=None,
        help="size of minibatches for testing only applies for flat-batch " + 
        "models. Default: None.")
    
    # Model configs    
    parser.add_argument("--runs", type=int, default=3, 
        help="number of runs to perform. Default: 3.")
    parser.add_argument("--train_iters", type=int, default=None,
        help="number of meta-training iterations. Default: None.")
    parser.add_argument("--eval_iters", type=int, default=600, 
        help="number of meta-valid/test iterations. Default: 600.")
    parser.add_argument("--val_after", type=int, default=2500,
        help="after how many episodes the meta-validation should be " +
        "performed. Default: 2500.")
    parser.add_argument("--seed", type=int, default=1337,
        help="random seed to use. Default: 1337.")
    parser.add_argument("--validate", action="store_true", default=True,
        help="validate performance on meta-validation tasks. Default: True.")
    parser.add_argument("--backbone", type=str, default="resnet18",
        help="backbone to use. Default: resnet18.")
    parser.add_argument("--freeze", action="store_true", default=True,
        help="whether to freeze the weights in the finetuning model of " +
        "earlier layers. Default: True.")
    parser.add_argument("--meta_batch_size", type=int, default=1,
        help="number of tasks to compute outer-update. Default: 1.")   
    parser.add_argument("--lr", type=float, default=None,
        help="learning rate for (meta-)optimizer. Default: None.")
    parser.add_argument("--T", type=int, default=None,
        help="number of weight updates per training set. Default: None.")
    parser.add_argument("--T_val", type=int, default=None,
        help="number of weight updates at validation time. Default: None.")
    parser.add_argument("--T_test", type=int, default=None,
        help="number of weight updates at test time. Default: None.")
    parser.add_argument("--base_lr", type=float, default=None,
        help="inner level learning rate: Default: None.")    
    parser.add_argument("--test_lr", type=float, default=0.001, 
        help="learning rate to use at meta-val/test time for finetuning. " +
        "Default: 0.001.")
    parser.add_argument("--test_opt", choices=["adam", "sgd"], default="adam",
        help="optimizer to use at meta-val/test time for finetuning. " +
        "Default: adam.")
    parser.add_argument("--img_size", type=int, default=128, 
        help="size of the images. Default: 128.")

    args, unparsed = parser.parse_known_args()
    
    return args, unparsed


class WithinDomainFewShotLearningExperiment:
    
    def __init__(self, args):
        self.args = args
        
        # Define paths
        self.curr_dir = os.path.dirname(__file__)
        self.main_dir = os.path.dirname(self.curr_dir)
        self.res_dir = os.path.join(self.main_dir, "Results")
        self.data_dir = os.path.join(self.main_dir, "Data")
        
        # Initialization step
        self.create_dirs()
        self.set_seed()
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.device = get_device(self.logs_path)
        self.gpu_info = get_torch_gpu_environment()
        self.clprint = lambda text: lprint(text, self.logs_path)
        self.clprint("\n".join(self.gpu_info)) 
        self.configure()
        
    def create_dirs(self):
        create_dir(self.res_dir)
        
        self.res_dir += "/within_domain_fsl/"
        create_dir(self.res_dir)
        
        self.res_dir += self.args.dataset + "/"
        create_dir(self.res_dir)
        
        if args.N: 
            self.res_dir += f"N{args.N}k{args.k}test{args.k_test}/"
        else: 
            self.res_dir += f"k{args.k}test{args.k_test}/" 
        create_dir(self.res_dir)
        
        self.res_dir += (self.args.model + 
            f"_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}" + "/")
        create_dir(self.res_dir)
        self.logs_path = f"{self.res_dir}logs.txt"
        
    def set_seed(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)

    def configure(self):
        # Mapping from model names to configurations
        mod_to_conf = {
            "tfs": (TrainFromScratch, deepcopy(TFS_CONF)),
            "finetuning": (FineTuning, deepcopy(FT_CONF)),
            "maml": (MAML, deepcopy(MAML_CONF)),
            "protonet": (PrototypicalNetwork, deepcopy(PROTO_CONF)),
            "matchingnet": (MatchingNetwork, deepcopy(MATCHING_CONF)) 
        }

        # Get model constructor and config for the specified algorithm
        self.model_constr, self.conf = mod_to_conf[self.args.model]

        # Set configurations
        self.overwrite_conf(self.conf, "train_batch_size")
        self.overwrite_conf(self.conf, "test_batch_size")
        self.overwrite_conf(self.conf, "T")
        self.overwrite_conf(self.conf, "lr")
        self.overwrite_conf(self.conf, "meta_batch_size")
        self.overwrite_conf(self.conf, "freeze")
        self.overwrite_conf(self.conf, "base_lr")
        
        if self.args.test_opt is not None or self.args.test_lr is not None:
            self.overwrite_conf(self.conf, "test_opt")
            self.overwrite_conf(self.conf, "test_lr")
        self.args.opt_fn = self.conf["opt_fn"]

        # Make sure argument 'val_after' is specified when 'validate'=True
        if self.args.validate:
            assert not self.args.val_after is None, "Please specify " + \
                "val_after (number of episodes after which to perform " + \
                "validation)"
                
        # If using multi-step maml, perform gradient clipping with -10, +10
        if "T" in self.conf:
            if self.conf["T"] > 1 and self.args.model=="maml":
                self.conf["grad_clip"] = 10
            else:
                self.conf["grad_clip"] = None
        
        if self.args.T_test is None:
            self.conf["T_test"] = self.conf["T"]
        else:
            self.conf["T_test"] = self.args.T_test
        
        if self.args.T_val is None:
            self.conf["T_val"] = self.conf["T"]
        else:
            self.conf["T_val"] = self.args.T_val

        self.conf["dev"] = self.device
        
        if self.args.N_train is None:
            self.args.N_train = self.args.N
        if self.args.k_train is None:
            self.args.k_train = self.args.k
        
        backbone_name = "resnet"
        num_blocks = int(self.args.backbone.split(backbone_name)[1])
        self.conf["baselearner_fn"] = ResNet

        if self.args.train_iters is None:
            self.args.train_iters = 60000 * self.conf["meta_batch_size"]

        (label_col, file_col, img_path, 
         md_path) = extract_info(self.data_dir, self.args.dataset)
        meta_data = pd.read_csv(md_path)
        classes = meta_data[label_col]

        # train, eval classes
        train_classes, eval_classes = train_test_split_classes(classes, 
            test_split=0.3)
        val_classes, test_classes = train_test_split_classes(eval_classes, 
            test_split=0.5)

        # if too little classes, use 60/20/20 split
        if len(val_classes) + len(test_classes) < 2*self.args.N:
            train_classes, eval_classes = train_test_split_classes(classes, 
                test_split=0.4)
            val_classes, test_classes = train_test_split_classes(eval_classes, 
                test_split=0.5)

        # backup
        if len(val_classes) <= self.args.N or len(test_classes) <= self.args.N:
            min_percentage = 2*(self.args.N+1) /len(np.unique(classes))
            train_classes, eval_classes = train_test_split_classes(classes, 
                test_split=min_percentage)
            val_classes, test_classes = train_test_split_classes(eval_classes, 
                test_split=0.5)
        del eval_classes 

        val_dataset = FewshotDataset(img_path=img_path, md_path=md_path, 
            label_col=label_col, file_col=file_col, allowed_labels=val_classes)
        val_sampler = CategoriesSampler(label=val_dataset.labels, 
            n_batch=self.args.eval_iters, n_cls=self.args.N, 
            n_per=self.args.k_train+self.args.k_test)
        self.val_loader = iter(self.cycle(DataLoader(dataset=val_dataset, 
            batch_sampler=val_sampler, num_workers=2, pin_memory=True)))

        test_dataset = FewshotDataset(img_path=img_path, md_path=md_path, 
            label_col=label_col, file_col=file_col, 
            allowed_labels=test_classes)
        test_sampler = CategoriesSampler(label=test_dataset.labels, 
            n_batch=self.args.eval_iters, n_cls=self.args.N, 
            n_per=self.args.k_train+self.args.k_test)
        self.test_loader = iter(self.cycle(DataLoader(dataset=test_dataset, 
            batch_sampler=test_sampler, num_workers=2, pin_memory=True)))

        if self.args.model in ("tfs", "finetuning"):
            self.args.batchmode = True
            if self.args.model == "tfs":
                num_train_classes = self.args.N_train
            else:
                num_train_classes = len(train_classes)

            # Batch data set
            train_dataset = FewshotDataset(img_path=img_path, md_path=md_path, 
                label_col=label_col, file_col=file_col, 
                allowed_labels=train_classes)
            self.train_loader = iter(self.cycle(DataLoader(
                dataset=train_dataset, batch_size=self.conf["train_batch_size"], 
                shuffle=True, num_workers=2)))
        else:
            self.args.batchmode = False
            num_train_classes = self.args.N_train
            train_dataset = FewshotDataset(img_path=img_path, md_path=md_path, 
                label_col=label_col, file_col=file_col, 
                allowed_labels=train_classes)
            train_sampler = CategoriesSampler(label=train_dataset.labels, 
                n_batch=self.args.train_iters, n_cls=self.args.N_train, 
                n_per=self.args.k_train+self.args.k_test)
            self.train_loader = iter(self.cycle(DataLoader(
                dataset=train_dataset, batch_sampler=train_sampler, 
                num_workers=2, pin_memory=True)))

        self.conf["baselearner_args"] = {
            "num_blocks": num_blocks,
            "dev": self.device,
            "train_classes": num_train_classes,
            "eval_classes": self.args.N, 
            "criterion": nn.CrossEntropyLoss(),
            "img_size": self.args.img_size
        }


        # Print the configuration for confirmation
        self.clprint("\n\n### ------------------------------------------ ###")
        self.clprint(f"Model: {self.args.model}")
        self.clprint(f"Training Classes: {len(train_classes)}")
        self.clprint(f"Validation Classes: {len(val_classes)}")
        self.clprint(f"Testing Classes: {len(test_classes)}")
        self.clprint(f"Random Seed: {self.args.seed}")
        self.print_conf()
        self.clprint("### ------------------------------------------ ###\n")
        
        self.save_experimental_settings()

        with open(f"{self.res_dir}config.pkl", "wb") as f:
            pickle.dump(self.conf, f)
            print(f"Stored config in {self.res_dir}config.pkl")

    def overwrite_conf(self, conf, arg_str):
        # If value provided in arguments, overwrite the config with it
        value = getattr(self.args, arg_str)
        if value is not None:
            conf[arg_str] = value
        else:
            if arg_str not in conf:
                conf[arg_str] = None
            else:
                setattr(self.args, arg_str, conf[arg_str]) 

    def cycle(self, iterable):
        while True:
            for x in iterable:
                yield x

    def print_conf(self):
        self.clprint(f"Configuration dump:")
        for k in self.conf.keys():
            self.clprint(f"\t{k} : {self.conf[k]}")

    def save_experimental_settings(self):
        join_list = lambda info: "\n".join(info)
        
        gpu_settings = ["\n--------------- GPU settings ---------------"]
        gpu_settings.extend(self.gpu_info)
        
        model_settings = [
            "\n--------------- Model settings ---------------",
            f"Model: {self.args.model}",
            f"Backbone: {self.args.backbone}",
            f"Random seed: {self.args.seed}",
            f"Loss function: CrossEntropy",
            f"Optimizer: {self.args.opt_fn.__name__}",
            f"Learning rate: {self.args.lr}",
            f"Training iterations: {self.args.train_iters}",
            f"Evaluation iterations: {self.args.eval_iters}",
            f"Validation active: {self.args.validate}",
            f"Validation after: {self.args.val_after}"
        ]
        
        data_settings = [
            "\n--------------- Data settings ---------------",
            f"Dataset: {self.args.dataset}",
            f"Image size: {self.args.img_size}"
        ]
        
        train_episodes_settings = [
            "\n--------------- Train episodes settings ---------------",
            f"N-way: {self.args.N_train}",
            f"k-shot: {self.args.k_train}",
            f"Query size: {self.args.k_test}",
            f"Batch size (FT): {self.args.train_batch_size}"
        ]
        
        test_episodes_settings = [
            "\n--------------- Evaluation episodes settings ---------------",
            f"N-way: {self.args.N}",
            f"k-shot: {self.args.k}",
            f"Query size: {self.args.k_test}",
            f"Batch size (TFS, FT): {self.args.test_batch_size}"
        ]
        
        model_specific_settings = [
            "\n--------------- Model specific settings ---------------",
            f"Meta batch size (PN, MN, MAML): {self.args.meta_batch_size}",
            f"Train inner steps (TFS, MAML): {self.args.T}",
            f"Validation inner steps (FT, MAML): {self.args.T_val}",
            f"Test inner steps (FT, MAML): {self.args.T_test}",
            f"Base learning rate (MAML): {self.args.base_lr}",
            f"Test optimizer (FT): {self.args.test_opt}",
            f"Test learning rate (FT): {self.args.test_lr}"
        ]
        
        all_settings = [
            "Description: This experiment is named 'Within-domain few-shot " +
            "learning'. It trains and test the algorithm with episodes which "+
            "are small classification tasks.",
            join_list(gpu_settings),
            join_list(model_settings),
            join_list(data_settings),
            join_list(train_episodes_settings),
            join_list(test_episodes_settings),
            join_list(model_specific_settings)
        ]

        experimental_settings_file = f"{self.res_dir}experimental_settings.txt"
        with open(experimental_settings_file, "w") as f:
            f.writelines(join_list(all_settings))

    def validate(self, model, val_loader, best_score, best_state):
        input_size = 0
        running_loss = 0
        running_corrects = 0
        scores = list()
        
        n_way = self.args.N
        k_shot = self.args.k_train
        query_size = self.args.k_test
        support_size = n_way * k_shot
        for i, task in enumerate(val_loader):
            data, labels = task
            
            labels = process_labels(n_way * (k_shot + query_size), n_way)
            train_x, train_y, test_x, test_y = (data[:support_size], 
                labels[:support_size], data[support_size:], 
                labels[support_size:])

            acc, loss_history, _, preds = model.evaluate(None, train_x, 
                train_y, test_x, test_y) 
            
            loss = loss_history[-1]
            labels = test_y.cpu().numpy()
            curr_input_size = len(test_y)
            
            input_size += curr_input_size
            running_loss += loss * curr_input_size
            running_corrects += np.sum(preds == labels)
            
            scores.append(acc)
            
            with open(self.val_results_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([acc, loss])
                
            if (i+1) == self.args.eval_iters:
                break                  

        val_loss = running_loss / input_size
        val_acc = running_corrects / input_size
        # Compute min/max (using model.operator) of new score and best score 
        tmp_score = max(val_acc, best_score)
        # There was an improvement, so store info
        if tmp_score != best_score and not math.isnan(tmp_score):
            best_score = val_acc
            best_state = model.dump_state()
            
        return best_score, best_state, val_acc, scores, val_loss
            
    def run(self):
        overall_best_score = -float("inf")
        overall_best_state = None
        
        seeds = [random.randint(0, 100000) for _ in range(self.args.runs)]
        print("Run seeds:", seeds)

        for run in range(self.args.runs):
            stime = time.time()
            self.run_res_dir = f"{self.res_dir}run{run}_seed{seeds[run]}/"
            create_dir(self.run_res_dir)
            
            VAL_SCORES = list()
            self.clprint("\n\n"+"-"*40)
            self.clprint(f"[*] Starting run {run} with seed {seeds[run]}")
            
            torch.manual_seed(seeds[run])
            model = self.model_constr(**self.conf)
            
            # Start with validation to ensure non-trainable model get 
            # validated at least once
            if self.args.validate:
                self.val_results_file =self.run_res_dir + \
                    "experiment_results_validation.csv"
                with open(self.val_results_file, "w+", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Accuracy", "Loss"])
                    
                best_score, best_state = model.init_score, None
                vtime = time.time()
                (best_score, best_state, val_acc, val_scores, 
                val_loss) = self.validate(model, self.val_loader, best_score, 
                    best_state)
                self.clprint("\nIteration: NA\tTrain loss: NA\tTrain acc: NA\t" +
                    "Training time: NA\t" +
                    f"Validation loss: {val_loss:.4f}\t" +
                    f"Validation acc: {val_acc:.4f}\t" +
                    f"Validation time: {time.time()-vtime} seconds") 
                VAL_SCORES.append(val_scores)
                # Stores all validation performances over time (learning curve) 
                learning_curve = [val_acc]
                
            allttime = time.time()
            n_way = self.args.N_train
            k_shot = self.args.k_train
            query_size = self.args.k_test
            support_size = n_way * k_shot
            if model.trainable:
                train_results_file = self.run_res_dir + \
                    "experiment_results_train.csv"
                with open(train_results_file, "w+", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Accuracy", "Loss"])
                    
                for i, task in enumerate(self.train_loader):
                    ttime = time.time()
                    if self.args.batchmode:
                        train_x, train_y = task
                        train_acc, train_loss, _,  _ = model.train(train_x, 
                            train_y.view(-1))
                    else:
                        data, labels = task
                        
                        labels = process_labels(n_way * (k_shot+query_size), 
                            n_way)
                        train_x, train_y, test_x, test_y = (data[:support_size], 
                            labels[:support_size], data[support_size:], 
                            labels[support_size:])
                        train_acc, train_loss, _, _ = model.train(train_x, 
                            train_y, test_x, test_y) 

                    ttime = time.time() - ttime

                    with open(train_results_file, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([train_acc, train_loss])
                        
                    # Perform meta-validation
                    if (self.args.validate and (i + 1) % (self.args.val_after * 
                        self.args.meta_batch_size) == 0):
                        vtime = time.time()
                        (best_score, best_state, val_acc, val_scores, 
                        val_loss) = self.validate(model, self.val_loader, 
                            best_score, best_state)
                        vtime = time.time() - vtime
                        VAL_SCORES.append(val_scores)
                        learning_curve.append(val_acc)
                        self.clprint(f"Iteration: {i+1}\t" +
                            f"Train loss: {train_loss:.4f}\t"+
                            f"Train acc: {train_acc:.4f}\t" +
                            f"Training time: {ttime} seconds\t" +
                            f"Validation loss: {val_loss:.4f}\t" +
                            f"Validation acc: {val_acc:.4f}\t" + 
                            f"Validation time: {vtime} seconds") 
                    else:
                        self.clprint(f"Iteration: {i+1}\t" +
                            f"Train loss: {train_loss:.4f}\t"+
                            f"Train acc: {train_acc:.4f}\t" +
                            f"Training time: {ttime} seconds")    
                    
                    if (i+1) == self.args.train_iters:
                        break

            self.clprint(f"\nTraining finshed after {time.time() - allttime} "
                + "seconds")

            if self.args.validate:
                self.clprint(f"Best validation acc: {best_score}")
                # Load best found state during meta-validation
                model.load_state(best_state)
                save_path = f"{self.run_res_dir}model.pkl"
                print(f"[*] Writing best model state to {save_path}")
                model.store_file(save_path)
                
                # Store validation accuracies
                vscore_file = f"{self.run_res_dir}val_accuracies.npy"
                VAL_SCORES = np.array(VAL_SCORES)
                np.save(vscore_file, VAL_SCORES)
                
                # Store learning curve
                self.clprint(f"Learning curve: {learning_curve}")
                curves_file = f"{self.run_res_dir}learning_curves" + \
                    f"{self.args.val_after}.csv"
                with open(curves_file, "w+", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([str(score) for score in learning_curve])
                
                # Check if the best score is better than the overall best score
                # if so, update best score and state across runs. 
                # It is better if tmp_best != best
                tmp_best_score = max(best_score, overall_best_score)
                if tmp_best_score != overall_best_score and not math.isnan(
                    tmp_best_score):
                    print(f"[*] Updated best model configuration across runs")
                    overall_best_score = best_score
                    overall_best_state = deepcopy(best_state)

            # Set seed and next test seed to ensure test diversity
            self.set_seed()
            
            test_results_file = f"{self.run_res_dir}experiment_results_test.csv"
            with open(test_results_file, "w+", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Accuracy", "Loss"])
            
            test_predictions_file = f"{self.run_res_dir}predictions_test.csv"
            with open(test_predictions_file, "w+", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Label", "Prediction", "Probabilities"])
                
            test_accuracies = list()
            self.clprint("\n[*] Evaluation...")
            n_way = self.args.N
            k_shot = self.args.k_train
            query_size = self.args.k_test
            support_size = n_way * k_shot
            for i, task in enumerate(self.test_loader):
                ttime = time.time()
                data, labels = task
                
                labels = process_labels(n_way * (k_shot + query_size), n_way)
                train_x, train_y, test_x, test_y = (data[:support_size], 
                    labels[:support_size], data[support_size:], 
                    labels[support_size:])
                
                acc, loss_history, probs, preds = model.evaluate(None, 
                    train_x, train_y, test_x, test_y, val=False)
                
                self.clprint(f"Iteration: {i+1}\t" +
                    f"Test loss: {loss_history[-1]:.4f}\t"+
                    f"Test acc: {acc:.4f}\t" +
                    f"Test time: {time.time() - ttime} seconds") 
                
                with open(test_results_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([acc, loss_history[-1]])
                    
                with open(test_predictions_file, "a", newline="") as f:
                    labels = test_y.cpu().numpy()
                    writer = csv.writer(f)
                    for j in range(len(labels)):
                        writer.writerow([labels[j], preds[j], list(probs[j])])
                
                test_accuracies.append(acc)
                if (i+1) == self.args.eval_iters:
                    break  
                
            # Store test scores
            test_scores_file = f"{self.res_dir}test_scores.csv"
            score_name = "accuracy"
            if run == 0:
                with open(test_scores_file, "w+", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["run", f"mean_{score_name}", 
                        f"median_{score_name}", "95ci", "time"])
            
            r, mean, median = (str(run), str(np.mean(test_accuracies)), 
                str(np.median(test_accuracies)))
            lb, _ = st.t.interval(alpha=0.95, df=len(test_accuracies)-1, 
                loc=np.mean(test_accuracies), scale=st.sem(test_accuracies)) 
            conf_interval = str(np.mean(test_accuracies) - lb)
            used_time = str(time.time() - stime)
            with open(test_scores_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([r, mean, median, conf_interval, used_time])

            self.clprint(f"\nRun {run} done, mean {score_name}: {mean}, " +
                f"median {score_name}: {median}, 95ci: {conf_interval}, " +
                f"time(s): {used_time}")
            self.clprint("-"*40)
            
        # At the end of all runs, write the best found configuration to file
        if self.args.validate:            
            save_path = f"{self.res_dir}best-model.pkl"
            print(f"[*] Writing best model state to {save_path}")
            model.load_state(overall_best_state)
            model.store_file(save_path)


if __name__ == "__main__":
    # Get args 
    args, unparsed = parse_arguments()

    # If there is still some unparsed argument, raise error
    if len(unparsed) != 0:
        raise ValueError(f"Argument {unparsed} not recognized")
    
    # Initiaize experiment object
    experiment = WithinDomainFewShotLearningExperiment(args)
    
    # Run experiment
    experiment.run()
    