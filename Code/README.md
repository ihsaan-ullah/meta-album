# **Meta-Album Code Reproducibility**
***

<br><br>

# Experiments

1. Continual Learning
2. Hierarchical Classification
3. Within-Domain Few-Shot Learning
4. Cross-Domain Few-Shot Learning

<br><br>

# Use Conda environment

Create env with python 3.8
```
conda create -n meta-album python=3.8
```

Activate env
```
conda activate meta-album
```

<br><br>

# Get ready for experiments

```
cd Code
```
install required packages
```
pip install -r requirements.txt
```

<br><br>

# Download data
Download data and put it in `Data` directory inside `Code`.  
Data should be in [Meta-Album Format](https://github.com/ihsaan-ullah/meta-album/tree/master/DataFormat)


<br><br>

# Run experiments

## Experiments

Go to scripts directory
```
cd Scripts
```

Launch the bash script corresponding for the experiment you are interested in.
```
bash continual_learning.sh
bash hierarchical_classification.sh
bash within_domain_few_shot_learning.sh
bash cross_domain_few_shot_learning.sh
```

## Results
The results will be stored in the following directories:
* `Results/continual_learning`
* `Results/hierarchical_classification`
* `Results/within_domain_fsl`
* `Results/cross_domain_fsl`














