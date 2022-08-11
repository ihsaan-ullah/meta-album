#!/bin/bash

# Scripts directory
Parent_Dir=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

# Trainers directory 
Trainers_Dir="$Parent_Dir/../Trainers"

echo $'############################################'
echo $'###--- Continual Learning Experiments ---###'
echo $'############################################\n\n'

Strategies=("Naive" "EWC" "Replay" "GEM" "AGEM" "Cumulative")
Seeds=(33 34 35)
Sequences=(0 1 2)

for strategy in ${Strategies[@]}; do 
    for seed in ${Seeds[@]}; do
        for seq in ${Sequences[@]}; do
            if [ "$strategy" = "Naive" ]; then
                python $Trainers_Dir/train_continual_learning.py --strategy $strategy --random_seed $seed --sequence_idx $seq --weight_decay 0.01
            fi
            python $Trainers_Dir/train_continual_learning.py --strategy $strategy --random_seed $seed --sequence_idx $seq
        done
    done
done
