#!/bin/bash

# For detailed instructions, check the main README.md file section Hierarchical Classification

# Scripts directory
Parent_Dir=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

# Trainers directory 
Trainers_Dir="$Parent_Dir/../Trainers"

# Datasets directory
Datasets_Dir="$Parent_Dir/../Data"

# Get all datasets from Datasets direcotry
# Assuming that only Hierarchical classification datasets are there in the directory


echo $'####################################################'
echo $'###--- Hierachical Classification Experiments ---###'
echo $'####################################################\n\n'


Datasets=`ls $Datasets_Dir`



if [[ ${Datasets[@]} ]]; then
    Seeds=(33 34 35)

    for seed in ${Seeds[@]}; do

        for Dataset in $Datasets; do
            # echo "#--- Dataset: $Dataset ---#" 
            echo
            python $Trainers_Dir/train_hierarchical_classification.py --dataset=$Dataset --random_seed=$seed
            echo 
            echo
    
        done
    done
else
    echo $"[-] No datasets found!"
fi














