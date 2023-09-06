#!/usr/bin/env bash

# nohup python3 prompt_input_generate_undirected_baseline_mlingual_BERT.py &
# wait


python3 ZeroEA_MASK_prompt_BERT_ALBERT_ELECTRA_ablation.py BERT > Results/Ablation_special_case_BERT_July21.txt &

python3 ZeroEA_MASK_prompt_BERT_ALBERT_ELECTRA_ablation.py ALBERT > Results/Ablation_special_case_ALBERT_July21.txt &

python3 ZeroEA_MASK_prompt_BERT_ALBERT_ELECTRA_ablation.py ELECTRA > Results/Ablation_special_case_ELECTRA_July21.txt &

wait

nohup python3 ZeroEA_MASK_prompt_roberta_BART_XLNet_ablation.py BART > Results/Ablation_special_case_BART_July21.txt &
nohup python3 ZeroEA_MASK_prompt_roberta_BART_XLNet_ablation.py XLNet > Results/Ablation_special_case_XLNet_July21.txt &

wait

nohup python3 ZeroEA_MASK_prompt_roberta_BART_XLNet_ablation.py xlm > Results/Ablation_special_case_xlmRoBERTa_July21.txt &
nohup python3 ZeroEA_MASK_prompt_roberta_BART_XLNet_ablation.py RoBERTa > Results/Ablation_special_case_RoBERTa_July21.txt &

wait

nohup python3 ZeroEA_MASK_prompt_T5_ablation.py > Results/Ablation_special_case_T5_July21.txt &
nohup python3 ZeroEA_MASK_prompt_GPT_ablation.py > Results/Ablation_special_case_gptj_July21.txt &