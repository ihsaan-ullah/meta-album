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
			# Test Set-0
			python $Trainers_Dir/train_cross_domain_fsl.py --model $model --n_way_eval 5 --k_shot_eval $shot --train_datasets DOG,INS_2,PLT_NET,MED_LF,PNU,RSICB,APL,TEX_DTD,ACT_40,MD_5_BIS --val_datasets AWA,INS,FNG,PLT_DOC,PRT,RSD,BTS,TEX_ALOT,ACT_410,MD_6 --test_datasets BRD,PLK,FLW,PLT_VIL,BCT,RESISC,CRS,TEX,SPT,MD_MIX --T 5 --T_val 10 --T_test 10
		
            # Test Set-1
			python $Trainers_Dir/train_cross_domain_fsl.py --model $model --n_way_eval 5 --k_shot_eval $shot --train_datasets AWA,INS,FNG,PLT_DOC,PRT,RSD,BTS,TEX_ALOT,ACT_410,MD_6 --val_datasets BRD,PLK,FLW,PLT_VIL,BCT,RESISC,CRS,TEX,SPT,MD_MIX --test_datasets DOG,INS_2,PLT_NET,MED_LF,PNU,RSICB,APL,TEX_DTD,ACT_40,MD_5_BIS --T 5 --T_val 10 --T_test 10
			
			# Test Set-2
			python $Trainers_Dir/train_cross_domain_fsl.py --model $model --n_way_eval 5 --k_shot_eval $shot --train_datasets BRD,PLK,FLW,PLT_VIL,BCT,RESISC,CRS,TEX,SPT,MD_MIX --val_datasets DOG,INS_2,PLT_NET,MED_LF,PNU,RSICB,APL,TEX_DTD,ACT_40,MD_5_BIS --test_datasets AWA,INS,FNG,PLT_DOC,PRT,RSD,BTS,TEX_ALOT,ACT_410,MD_6 --T 5 --T_val 10 --T_test 10
        else
			# Test Set-0
			python $Trainers_Dir/train_cross_domain_fsl.py --model $model --n_way_eval 5 --k_shot_eval $shot --train_datasets DOG,INS_2,PLT_NET,MED_LF,PNU,RSICB,APL,TEX_DTD,ACT_40,MD_5_BIS --val_datasets AWA,INS,FNG,PLT_DOC,PRT,RSD,BTS,TEX_ALOT,ACT_410,MD_6 --test_datasets BRD,PLK,FLW,PLT_VIL,BCT,RESISC,CRS,TEX,SPT,MD_MIX
		
            # Test Set-1
			python $Trainers_Dir/train_cross_domain_fsl.py --model $model --n_way_eval 5 --k_shot_eval $shot --train_datasets AWA,INS,FNG,PLT_DOC,PRT,RSD,BTS,TEX_ALOT,ACT_410,MD_6 --val_datasets BRD,PLK,FLW,PLT_VIL,BCT,RESISC,CRS,TEX,SPT,MD_MIX --test_datasets DOG,INS_2,PLT_NET,MED_LF,PNU,RSICB,APL,TEX_DTD,ACT_40,MD_5_BIS
			
			# Test Set-2
			python $Trainers_Dir/train_cross_domain_fsl.py --model $model --n_way_eval 5 --k_shot_eval $shot --train_datasets BRD,PLK,FLW,PLT_VIL,BCT,RESISC,CRS,TEX,SPT,MD_MIX --val_datasets DOG,INS_2,PLT_NET,MED_LF,PNU,RSICB,APL,TEX_DTD,ACT_40,MD_5_BIS --test_datasets AWA,INS,FNG,PLT_DOC,PRT,RSD,BTS,TEX_ALOT,ACT_410,MD_6
        fi
    done
    
    # Any-way any-shot
    if [ "$model" = "maml" ]; then
		# Test Set-0
		python $Trainers_Dir/train_cross_domain_fsl.py --model $model --train_datasets DOG,INS_2,PLT_NET,MED_LF,PNU,RSICB,APL,TEX_DTD,ACT_40,MD_5_BIS --val_datasets AWA,INS,FNG,PLT_DOC,PRT,RSD,BTS,TEX_ALOT,ACT_410,MD_6 --test_datasets BRD,PLK,FLW,PLT_VIL,BCT,RESISC,CRS,TEX,SPT,MD_MIX --T 5 --T_val 10 --T_test 10
	
		# Test Set-1
		python $Trainers_Dir/train_cross_domain_fsl.py --model $model --train_datasets AWA,INS,FNG,PLT_DOC,PRT,RSD,BTS,TEX_ALOT,ACT_410,MD_6 --val_datasets BRD,PLK,FLW,PLT_VIL,BCT,RESISC,CRS,TEX,SPT,MD_MIX --test_datasets DOG,INS_2,PLT_NET,MED_LF,PNU,RSICB,APL,TEX_DTD,ACT_40,MD_5_BIS --T 5 --T_val 10 --T_test 10
		
		# Test Set-2
		python $Trainers_Dir/train_cross_domain_fsl.py --model $model --train_datasets BRD,PLK,FLW,PLT_VIL,BCT,RESISC,CRS,TEX,SPT,MD_MIX --val_datasets DOG,INS_2,PLT_NET,MED_LF,PNU,RSICB,APL,TEX_DTD,ACT_40,MD_5_BIS --test_datasets AWA,INS,FNG,PLT_DOC,PRT,RSD,BTS,TEX_ALOT,ACT_410,MD_6 --T 5 --T_val 10 --T_test 10
    else
		# Test Set-0
		python $Trainers_Dir/train_cross_domain_fsl.py --model $model --train_datasets DOG,INS_2,PLT_NET,MED_LF,PNU,RSICB,APL,TEX_DTD,ACT_40,MD_5_BIS --val_datasets AWA,INS,FNG,PLT_DOC,PRT,RSD,BTS,TEX_ALOT,ACT_410,MD_6 --test_datasets BRD,PLK,FLW,PLT_VIL,BCT,RESISC,CRS,TEX,SPT,MD_MIX
	
		# Test Set-1
		python $Trainers_Dir/train_cross_domain_fsl.py --model $model --train_datasets AWA,INS,FNG,PLT_DOC,PRT,RSD,BTS,TEX_ALOT,ACT_410,MD_6 --val_datasets BRD,PLK,FLW,PLT_VIL,BCT,RESISC,CRS,TEX,SPT,MD_MIX --test_datasets DOG,INS_2,PLT_NET,MED_LF,PNU,RSICB,APL,TEX_DTD,ACT_40,MD_5_BIS
		
		# Test Set-2
		python $Trainers_Dir/train_cross_domain_fsl.py --model $model --train_datasets BRD,PLK,FLW,PLT_VIL,BCT,RESISC,CRS,TEX,SPT,MD_MIX --val_datasets DOG,INS_2,PLT_NET,MED_LF,PNU,RSICB,APL,TEX_DTD,ACT_40,MD_5_BIS --test_datasets AWA,INS,FNG,PLT_DOC,PRT,RSD,BTS,TEX_ALOT,ACT_410,MD_6
    fi
done
