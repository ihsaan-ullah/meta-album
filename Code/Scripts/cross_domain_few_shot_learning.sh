#!/bin/bash

# Scripts directory
Parent_Dir=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

# Trainers directory 
Trainers_Dir="$Parent_Dir/../Trainers"

echo $'########################################################'
echo $'###--- Cross-Domain Few-Shot Learning Experiments ---###'
echo $'########################################################\n\n'

Models=("tfs" "finetuning" "maml" "protonet" "matchingnet")

for model in ${Models[@]}; do
    # Fixed ways and shots
    Shots=(1 5 10 20)
    for shot in ${Shots[@]}; do
        if [ "$model" = "maml" ]; then
            python $Trainers_Dir/train_cross_domain_fsl.py --model $model --n_way_eval 5 --k_shot_eval $shot --train_datasets BCT,BRD,CRS --val_datasets FLW,MD_MIX,PLK --test_datasets PLT_VIL,RESISC,SPT,TEX --T 5 --T_val 10 --T_test 10
        else
            python $Trainers_Dir/train_cross_domain_fsl.py --model $model --n_way_eval 5 --k_shot_eval $shot --train_datasets BCT,BRD,CRS --val_datasets FLW,MD_MIX,PLK --test_datasets PLT_VIL,RESISC,SPT,TEX
        fi
    done
    
    # Any-way any-shot
    if [ "$model" = "maml" ]; then
        python $Trainers_Dir/train_cross_domain_fsl.py --model $model --train_datasets BCT,BRD,CRS --val_datasets FLW,MD_MIX,PLK --test_datasets PLT_VIL,RESISC,SPT,TEX --T 5 --T_val 10 --T_test 10
    else
        python $Trainers_Dir/train_cross_domain_fsl.py --model $model --train_datasets BCT,BRD,CRS --val_datasets FLW,MD_MIX,PLK --test_datasets PLT_VIL,RESISC,SPT,TEX
    fi
done
