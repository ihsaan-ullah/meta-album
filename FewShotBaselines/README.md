# Meta-Album

This directory contains all code that can be used to reproduce (and extend) the few-shot learning results in the paper. 

## Setup
Steps to get code to run:


1. Download the *publicdata.zip* file (https://drive.google.com/file/d/1fBvpb_E9FBsVwJRl8OsFZdJ9f_9cOJZl/view?usp=sharing) and place it **AS ZIP (DO NOT UNPACK YET THAT WILL BE DONE AUTOMATICALLY!)** into the meta-album/FewShotBaselines directory
2. Change the current directory to this directory ```cd meta-album/FewShotBaselines/```
3. Install all requirements from the *reqs.txt* file (in your Conda environment) using ```pip install -r reqs.txt```
4. Run the setup script that unpacks all data sets using: ```python  setup_data.py```
5. You are now all set! You can now start running experiments!

If you have trouble installing dependencies, these instructions may help:

Create a new conda environment, then\
`conda install python=3.7 pip=20.2.4 -y`

In reqs.txt, remove version number of these packages, line 65-67 \
mkl-fft\
mkl-random\
mkl-service


## Running experiments
You can use the commands below (when you are in ./meta-album) to run experiments. Make sure to **substitute <DATASET> for the correct dataset identifier and <NUM_SHOTS> for the correct number of support examples per class.** 
  
  
  
### TrainFromScratch
```python -u main.py --problem <DATASET> --k <NUM_SHOTS> --backbone resnet18 --N 5 --k_test 16 --validate --val_after 2500 --model tfs --runs 3```

### Finetuning
```python -u main.py --problem <DATASET> --k <NUM_SHOTS> --backbone resnet18 --N 5 --k_test 16 --validate --val_after 2500 --model finetuning --runs 3```
  
### MAML 
```python -u main.py --problem <DATASET> --k <NUM_SHOTS> --model maml --backbone resnet18 --k_test 16 --N 5 --model maml --val_after 2500 --lr 0.001 --T 5 --meta_batch_size 2 --runs 3 --validate```

### Matching Networks
```python -u main.py --problem <DATASET> --model matchingnet --k <NUM_SHOTS> --backbone resnet18 --N 5 --k_test 16 --val_after 2500 --validate --runs 3```

### Prototypical Networks
```python -u main.py --problem <DATASET> --model protonet --k <NUM_SHOTS> --backbone resnet18 --N 5 --k_test 16 --val_after 2500 --validate --runs 3```


## Dataset specifiers
  The following arguments can be used as <DATASET> in the above scripts in the shape identifier : name in the paper. 
  - min : miniImageNet (used for verifying implementations)
  - cub : CUB (not used in paper)
  - insects : Insects  
  - plankton : Plankton
  - plants : Mini Plant Village
  - medleaf : Medical Leaf
  - texture1 : Textures
  - texture2 : Textures DTD
  - rsicb : Mini RSICB
  - resisc: Mini RESISC
  - omniprint1 : OmniPrint-MD-mix
  - omniprint2 : OmniPrint-MD-5-bis
  
## Creating your own algorithm

To create your own algorithm, you can create a new file called youralgorithm.py in the algorithms folder. In that file, you can define your own algorithm class which inherits from the Algorithm class specified in algorithm.py. You will need to implement 3 functions: init (the initialization function), train (train on a given task), and evaluate (apply your algorithm to the task and obtain the performance on the query set). You can use other defind algorithms as examples on how to do this.

Once you have done this, you can define default arguments for the init function in configs.py. Lastly, you can import your config, add a string identifier for your algorithm to the choices field in the --model argument in main.py, and add it to the dictionary mod_to_conf. You are then fully set to run your new algorithm by calling main.py with the argument --model youralgorithmspecifier!
  
  
## Contributing results
If you would like to contribute an algorithm, please create a pull request.  





