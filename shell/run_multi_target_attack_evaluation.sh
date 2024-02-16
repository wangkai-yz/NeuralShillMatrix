#!/bin/bash

target_ids_list=("1689" "2001")

for i in "${!target_ids_list[@]}"; do
  target_ids="${target_ids_list[$i]}"

  for model in NNMF IAutoRec; do
	  echo "$i $target_ids 评估攻击 ${model}"
    python3.8 ../run_main/run_evaluation_model.py --dataset_name filmTrust --attack_method gan,BigGan,average,segment,random,bandwagon --model_name ${model} --targets "${target_ids}" --multiple_objectives 1 --attack_count 50 --filler_count 16
	done

done

