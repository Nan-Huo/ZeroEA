#!bin/bash

# PLM
echo "Start generating PLM input prompts"
python3 ZeroEA_input_generate_undirected.py  ../data/DBP15K/zh_en/ text_input_no_train_11_wxt.txt text_input_no_train_22_wxt.txt False # you can change the data dir; input file dirs; WebSearch use flag as you like

echo "Start encoding and EA"
python3 ZeroEA_base.py ../data/DBP15K/zh_en/text_input_no_train_11_wxt.txt ../data/DBP15K/zh_en/text_input_no_train_22_wxt.txt # you can change the input file dirs as you like


# PLM + Tool
echo "Start generating PLM input prompts USING TOOL"
python3 ZeroEA_input_generate_undirected.py  ../data/DBP15K/zh_en/ text_input_no_train_11_wxt_KI.txt text_input_no_train_22_wxt_KI.txt True # you can change the data dir; input file dirs; WebSearch use flag as you like

echo "Start encoding and EA USING TOOL"
python3 ZeroEA_use_tool.py 80 ../data/DBP15K/zh_en/text_input_no_train_11_wxt_KI.txt ../data/DBP15K/zh_en/text_input_no_train_22_wxt_KI.txt ../data/DBP15K/zh_en/text_input_no_train_11_wxt.txt ../data/DBP15K/zh_en/text_input_no_train_22_wxt.txt# you can change the Rouge threshold; input file dirs as you like
