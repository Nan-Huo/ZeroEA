#!/usr/bin/env bash

# Generation
nohup python3 prompt_input_generate_undirected_baseline_motif_2hop_KI.py 3 &
nohup python3 prompt_input_generate_undirected_baseline_motif_2hop_KI.py 5 &
nohup python3 prompt_input_generate_undirected_baseline_motif_2hop_KI.py 8 &
nohup python3 prompt_input_generate_undirected_baseline_motif_2hop_KI.py 10 &
nohup python3 prompt_input_generate_undirected_baseline_motif_2hop_KI.py 12 &
nohup python3 prompt_input_generate_undirected_baseline_motif_2hop_KI.py 15 &
nohup python3 prompt_input_generate_undirected_baseline_motif_2hop_KI.py 18 &
wait

# Measurement
nohup python3 Prompt_EA_no_train_MASK_prompt_Euc_Cos_KI.py text_input_no_train_11_wxt_2hop_motif_KI_3.txt text_input_no_train_22_wxt_2hop_motif_KI_3.txt > Results_KI_2hop/Output_motif_2hop_KI_neighbor_3.txt &

nohup python3 Prompt_EA_no_train_MASK_prompt_Euc_Cos_KI.py text_input_no_train_11_wxt_2hop_motif_KI_5.txt text_input_no_train_22_wxt_2hop_motif_KI_5.txt > Results_KI_2hop/Output_motif_2hop_KI_neighbor_5.txt &
wait

nohup python3 Prompt_EA_no_train_MASK_prompt_Euc_Cos_KI.py text_input_no_train_11_wxt_2hop_motif_KI_8.txt text_input_no_train_22_wxt_2hop_motif_KI_8.txt > Results_KI_2hop/Output_motif_2hop_KI_neighbor_8.txt &

nohup python3 Prompt_EA_no_train_MASK_prompt_Euc_Cos_KI.py text_input_no_train_11_wxt_2hop_motif_KI_10.txt text_input_no_train_22_wxt_2hop_motif_KI_10.txt > Results_KI_2hop/Output_motif_2hop_KI_neighbor_10.txt &
wait

nohup python3 Prompt_EA_no_train_MASK_prompt_Euc_Cos_KI.py text_input_no_train_11_wxt_2hop_motif_KI_12.txt text_input_no_train_22_wxt_2hop_motif_KI_12.txt > Results_KI_2hop/Output_motif_2hop_KI_neighbor_12.txt &

nohup python3 Prompt_EA_no_train_MASK_prompt_Euc_Cos_KI.py text_input_no_train_11_wxt_2hop_motif_KI_15.txt text_input_no_train_22_wxt_2hop_motif_KI_15.txt > Results_KI_2hop/Output_motif_2hop_KI_neighbor_15.txt &

nohup python3 Prompt_EA_no_train_MASK_prompt_Euc_Cos_KI.py text_input_no_train_11_wxt_2hop_motif_KI_18.txt text_input_no_train_22_wxt_2hop_motif_KI_18.txt > Results_KI_2hop/Output_motif_2hop_KI_neighbor_18.txt &

wait

# ---------------------------------------------------------------------------
# Generation
nohup python3 prompt_input_generate_undirected_baseline_motif_2hop.py 3 &
nohup python3 prompt_input_generate_undirected_baseline_motif_2hop.py 5 &
nohup python3 prompt_input_generate_undirected_baseline_motif_2hop.py 8 &
nohup python3 prompt_input_generate_undirected_baseline_motif_2hop.py 10 &
nohup python3 prompt_input_generate_undirected_baseline_motif_2hop.py 12 &
nohup python3 prompt_input_generate_undirected_baseline_motif_2hop.py 15 &
nohup python3 prompt_input_generate_undirected_baseline_motif_2hop.py 18 &
wait

# Measurement
nohup python3 Prompt_EA_no_train_MASK_prompt_Euc_Cos_KI.py text_input_no_train_11_wxt_2hop_motif_3.txt text_input_no_train_22_wxt_2hop_motif_3.txt > Results_noKI_2hop/Output_motif_2hop_neighbor_3.txt &

nohup python3 Prompt_EA_no_train_MASK_prompt_Euc_Cos_KI.py text_input_no_train_11_wxt_2hop_motif_5.txt text_input_no_train_22_wxt_2hop_motif_5.txt > Results_noKI_2hop/Output_motif_2hop_neighbor_5.txt &
wait

nohup python3 Prompt_EA_no_train_MASK_prompt_Euc_Cos_KI.py text_input_no_train_11_wxt_2hop_motif_8.txt text_input_no_train_22_wxt_2hop_motif_8.txt > Results_noKI_2hop/Output_motif_2hop_neighbor_8.txt &

nohup python3 Prompt_EA_no_train_MASK_prompt_Euc_Cos_KI.py text_input_no_train_11_wxt_2hop_motif_10.txt text_input_no_train_22_wxt_2hop_motif_10.txt > Results_noKI_2hop/Output_motif_2hop_neighbor_10.txt &
wait

nohup python3 Prompt_EA_no_train_MASK_prompt_Euc_Cos_KI.py text_input_no_train_11_wxt_2hop_motif_12.txt text_input_no_train_22_wxt_2hop_motif_12.txt > Results_noKI_2hop/Output_motif_2hop_neighbor_12.txt &

nohup python3 Prompt_EA_no_train_MASK_prompt_Euc_Cos_KI.py text_input_no_train_11_wxt_2hop_motif_15.txt text_input_no_train_22_wxt_2hop_motif_15.txt > Results_noKI_2hop/Output_motif_2hop_neighbor_15.txt &

nohup python3 Prompt_EA_no_train_MASK_prompt_Euc_Cos_KI.py text_input_no_train_11_wxt_2hop_motif_18.txt text_input_no_train_22_wxt_2hop_motif_18.txt > Results_noKI_2hop/Output_motif_2hop_neighbor_18.txt &


# # Generation
# nohup python3 prompt_input_generate_undirected_baseline_motif_1hop_KI.py 1 &
# nohup python3 prompt_input_generate_undirected_baseline_motif_1hop_KI.py 2 &
# nohup python3 prompt_input_generate_undirected_baseline_motif_1hop_KI.py 3 &
# nohup python3 prompt_input_generate_undirected_baseline_motif_1hop_KI.py 4 &
# nohup python3 prompt_input_generate_undirected_baseline_motif_1hop_KI.py 6 &
# nohup python3 prompt_input_generate_undirected_baseline_motif_1hop_KI.py 7 &
# nohup python3 prompt_input_generate_undirected_baseline_motif_1hop_KI.py 8 &
# wait

# # Measurement
# nohup python3 Prompt_EA_no_train_MASK_prompt_Euc_Cos_KI.py text_input_no_train_11_wxt_motif_KI_1.txt text_input_no_train_22_wxt_motif_KI_1.txt > Results_KI/Output_motif_1hop_KI_neighbor_1.txt &

# nohup python3 Prompt_EA_no_train_MASK_prompt_Euc_Cos_KI.py text_input_no_train_11_wxt_motif_KI_2.txt text_input_no_train_22_wxt_motif_KI_2.txt > Results_KI/Output_motif_1hop_KI_neighbor_2.txt &
# wait

# nohup python3 Prompt_EA_no_train_MASK_prompt_Euc_Cos_KI.py text_input_no_train_11_wxt_motif_KI_3.txt text_input_no_train_22_wxt_motif_KI_3.txt > Results_KI/Output_motif_1hop_KI_neighbor_3.txt &

# nohup python3 Prompt_EA_no_train_MASK_prompt_Euc_Cos_KI.py text_input_no_train_11_wxt_motif_KI_4.txt text_input_no_train_22_wxt_motif_KI_4.txt > Results_KI/Output_motif_1hop_KI_neighbor_4.txt &
# wait

# nohup python3 Prompt_EA_no_train_MASK_prompt_Euc_Cos_KI.py text_input_no_train_11_wxt_motif_KI_6.txt text_input_no_train_22_wxt_motif_KI_6.txt > Results_KI/Output_motif_1hop_KI_neighbor_6.txt &

# nohup python3 Prompt_EA_no_train_MASK_prompt_Euc_Cos_KI.py text_input_no_train_11_wxt_motif_KI_7.txt text_input_no_train_22_wxt_motif_KI_7.txt > Results_KI/Output_motif_1hop_KI_neighbor_7.txt &

# nohup python3 Prompt_EA_no_train_MASK_prompt_Euc_Cos_KI.py text_input_no_train_11_wxt_motif_KI_8.txt text_input_no_train_22_wxt_motif_KI_8.txt > Results_KI/Output_motif_1hop_KI_neighbor_8.txt &

# wait

# # ---------------------------------------------------------------------------
# # Generation
# nohup python3 prompt_input_generate_undirected_baseline_motif_1hop.py 1 &
# nohup python3 prompt_input_generate_undirected_baseline_motif_1hop.py 2 &
# nohup python3 prompt_input_generate_undirected_baseline_motif_1hop.py 3 &
# nohup python3 prompt_input_generate_undirected_baseline_motif_1hop.py 4 &
# nohup python3 prompt_input_generate_undirected_baseline_motif_1hop.py 6 &
# nohup python3 prompt_input_generate_undirected_baseline_motif_1hop.py 7 &
# nohup python3 prompt_input_generate_undirected_baseline_motif_1hop.py 8 &
# wait

# # Measurement
# nohup python3 Prompt_EA_no_train_MASK_prompt_Euc_Cos_KI.py text_input_no_train_11_wxt_motif_1.txt text_input_no_train_22_wxt_motif_1.txt > Results_noKI/Output_motif_1hop_neighbor_1.txt &

# nohup python3 Prompt_EA_no_train_MASK_prompt_Euc_Cos_KI.py text_input_no_train_11_wxt_motif_2.txt text_input_no_train_22_wxt_motif_2.txt > Results_noKI/Output_motif_1hop_neighbor_2.txt &
# wait

# nohup python3 Prompt_EA_no_train_MASK_prompt_Euc_Cos_KI.py text_input_no_train_11_wxt_motif_3.txt text_input_no_train_22_wxt_motif_3.txt > Results_noKI/Output_motif_1hop_neighbor_3.txt &

# nohup python3 Prompt_EA_no_train_MASK_prompt_Euc_Cos_KI.py text_input_no_train_11_wxt_motif_4.txt text_input_no_train_22_wxt_motif_4.txt > Results_noKI/Output_motif_1hop_neighbor_4.txt &
# wait

# nohup python3 Prompt_EA_no_train_MASK_prompt_Euc_Cos_KI.py text_input_no_train_11_wxt_motif_6.txt text_input_no_train_22_wxt_motif_6.txt > Results_noKI/Output_motif_1hop_neighbor_6.txt &

# nohup python3 Prompt_EA_no_train_MASK_prompt_Euc_Cos_KI.py text_input_no_train_11_wxt_motif_7.txt text_input_no_train_22_wxt_motif_7.txt > Results_noKI/Output_motif_1hop_neighbor_7.txt &

# nohup python3 Prompt_EA_no_train_MASK_prompt_Euc_Cos_KI.py text_input_no_train_11_wxt_motif_8.txt text_input_no_train_22_wxt_motif_8.txt > Results_noKI/Output_motif_1hop_neighbor_8.txt &