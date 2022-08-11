#!/bin/bash

# Scripts directory
Parent_Dir=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

# Trainers directory 
Trainers_Dir="$Parent_Dir/../Trainers"

echo $'########################################################'
echo $'###--- Within-Domain Few-Shot Learning Experiments ---###'
echo $'########################################################\n\n'

Datasets=("BCT" "BRD" "CRS" "FLW" "MD_MIX" "PLK" "PLT_VIL" "RESISC" "SPT" "TEX")
Models=("tfs" "finetuning" "maml" "protonet" "matchingnet")
Shots=(1 5 10 20)

for dataset in ${Datasets[@]}; do 
    for model in ${Models[@]}; do
        for shot in ${Shots[@]}; do
            if [ "$model" = "maml" ]; then
                python $Trainers_Dir/train_within_domain_fsl.py --dataset $dataset --model $model --k $shot --T 5 --T_val 10 --T_test 10
            else
                python $Trainers_Dir/train_within_domain_fsl.py --dataset $dataset --model $model --k $shot
            fi
        done
    done
done