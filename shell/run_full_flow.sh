#!/bin/bash

bandwagon_selected_1808="11,16,214"
bandwagon_selected_1689="1,6,204"
bandwagon_selected_1691="6,7,214"
bandwagon_selected_2001="1,10,16"
bandwagon_selected_1959="10,4,12"

target_ids_list=("1808" "1689" "1691" "2001" "1959")
#target_ids_list=("1808" "1689")

bandwagon_selected_list=("$bandwagon_selected_1689" "$bandwagon_selected_1691" "$bandwagon_selected_2001" "$bandwagon_selected_1959")

for i in "${!target_ids_list[@]}"; do
  target_ids="${target_ids_list[$i]}"
  bandwagon_selected="${bandwagon_selected_list[$i]}"

  echo "$i $target_ids and bandwagon selected: $bandwagon_selected"

  #生成攻击设置数据
  echo "$i $target_ids 生成攻击设置数据"
  python3.8 ../run_main/run_initialization_attack.py --dataset_name filmTrust --targets "${target_ids}" --attack_count 50 --filler_count 16
  #进行基线攻击
  echo "$i $target_ids 进行基线攻击"
  python3.8 ../run_main/run_baseline_attack.py --dataset_name filmTrust --attack_methods average,segment,random,bandwagon --targets "${target_ids}" --attack_count 50 --filler_count 16 --bandwagon_selected "${bandwagon_selected}"
  #进行gan攻击
  echo "$i $target_ids 进行gan攻击"
  python3.8 ../run_main/run_gan_attack.py --dataset_name filmTrust --targets "${target_ids}" --attack_methods gan --attack_count 50 --filler_count 16

  for model in NNMF IAutoRec; do
    #推荐系统评估基线攻击
    echo "$i $target_ids 评估基线攻击 ${model}"
    python3.8 ../run_main/run_evaluation_model.py --dataset_name filmTrust --attack_methods average,segment,random,bandwagon --model_name "${model}" --targets "${target_ids}" --attack_count 50 --filler_count 16
	  #评估攻击前的推荐系统
    echo "$i $target_ids 评估攻击前的推荐系统 ${model}"
    python3.8 ../run_main/run_evaluation_model.py --dataset_name filmTrust --attack_method no --model_name ${model} --targets "${target_ids}" --attack_count 50 --filler_count 16

	  echo "$i $target_ids 评估gan攻击 ${model}"
    python3.8 ../run_main/run_evaluation_model.py --dataset_name filmTrust --attack_method gan --model_name ${model} --targets "${target_ids}" --attack_count 50 --filler_count 16
	done

done

