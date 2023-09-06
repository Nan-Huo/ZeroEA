# ZeroEA: A Zero-Training Entity Alignment Framework via Pre-Trained Language Model

[![Python 3.7+](https://img.shields.io/badge/Python-3.7+-teal.svg)](https://www.python.org/downloads/release/python-390/)
[![Pytorch 1.8+](https://img.shields.io/badge/Pytorch-1.8+-red.svg)](https://pytorch.org/blog/pytorch-1.8-released/)
[![Transformers 4.8+](https://img.shields.io/badge/Transformers-4.8+-cine.svg)](https://pypi.org/project/transformers/)
[![BERT 1.8+](https://img.shields.io/badge/BERT-PLM-blue.svg)](https://huggingface.co/bert-base-uncased)
[![DBP15K 1.8+](https://img.shields.io/badge/DBP15K-benchmark-green.svg)](https://drive.google.com/file/d/1Now8iTn37QYMOUC80swlBq9QKxKhFmSU/view)
[![DWY100K 1.8+](https://img.shields.io/badge/DWY100K-benchmark-orange.svg)](https://github.com/nju-websoft/BootEA/tree/master/dataset/DWY100K)
[![Leaderboard 1.8+](https://img.shields.io/badge/SPIDER-benchmark-pink.svg)](https://yale-lily.github.io/spider)

<p align="center" width="100%">
<a><img src="img/ZeroEA.png" style="width: 100%; min-width: 300px; display: block; margin: auto;"></a>
</p>


## Overview
ZeroEA is a novel **zero-training** entity alignment framework for knowledge graphs. It bridges the gap between graph structure and plain text by converting KG topology into textual context suitable for PLM input. Additionally, in order to provide PLMs with concise and clear input text of reasonable length, we design a motif-based neighborhood
filter to eliminate noisy neighbors. Notably, ZeroEA can **outperform state-of-the-art supervised baselines**, and our study highlights the considerable potential of EA technique in improving the performance of downstream tasks, thereby benefitting the broader research field.


## Installation

### Create environment and download dependencies
• Please create the virtual environment through
```bash
conda create -n zeroea python=3.6
source activate zeroea
```
• Also, download dependencies:
```txt
torch==1.9.0
numpy==1.19.2
sklearn==1.3.0
transformers==4.8.2
torchtext==0.10.0
```

## Quick Start

### Data Preparation

You can download the DBP15K data from [here](https://drive.google.com/file/d/1Now8iTn37QYMOUC80swlBq9QKxKhFmSU/view) and DWY100K data from [here](https://github.com/nju-websoft/BootEA/tree/master/dataset/DWY100K).

### Directory structure
Take DBP15K (ZH-EN) as an example, the folder "zh_en" contains:
* all_attrs_range: the range code of attributes in source KG (ZH);
* ent_ILLs: all entity links (15K);
* rel_ILLs: all relationship links (with the same URI or localname);
* s_labels: cross-lingual entity labels of source KG (ZH);
* s_triples: relationship triples of source KG (ZH);
* sup_attr_pairs: all attribute links (with the same URI or localname);
* t_labels: cross-lingual entity labels of target KG (EN);
* t_triples: relationship triples of target KG (EN);
* training_attrs_1: entity attributes in source KG (ZH);
* training_attrs_2: entity attributes in target KG (EN);

### Dataset files
Take the dataset "0_3" of DBP15K (ZH-EN) as an example, the folder "0_3" (means 30% training data split) contains:
* ent_ids_1: ids for entities in source KG (ZH);
* ent_ids_2: ids for entities in target KG (EN);
* ref_ent_ids: entity links encoded by ids for testing;
* ref_ents: URIs of entity links for testing;
* rel_ids_1: ids for relationships in source KG (ZH);
* rel_ids_2: ids for relationships in target KG (EN);
* sup_ent_ids: entity links encoded by ids for training;
* sup_rel_ids: relationship links encoded by ids for training;
* triples_1: relationship triples encoded by ids in source KG (ZH);
* triples_2: relationship triples encoded by ids in target KG (EN);

### Run Experiments

**To run ZeroEA without tool use**, please first generate the discrete prompts as the input of BERT:
```bash
python3 ZeroEA_input_generate_undirected  ../data/DBP15K/zh_en/ text_input_no_train_11_wxt.txt text_input_no_train_22_wxt.txt False
```
You can change the file directory at the first parameter; the file names of the prompts of two KGs at the second and third parameters; and the tool use flag at the last parameter. Then run the encoding & eval code:
```bash
python3 ZeroEA_base.py ../data/DBP15K/zh_en/text_input_no_train_11_wxt.txt ../data/DBP15K/zh_en/text_input_no_train_22_wxt.txt
```
where the two parameters are the input prompts files directory.  


**To run ZeroEA with tool use**, please first generate the discrete prompts as the input of BERT:
```bash
python3 ZeroEA_input_generate_undirected  ../data/DBP15K/zh_en/ text_input_no_train_11_wxt_KI.txt text_input_no_train_22_wxt_KI.txt True
```
You can change the file directory at the first parameter; the file names of the prompts of two KGs at the second and third parameters; and the tool use flag at the last parameter. Then run the encoding & eval code:
```bash
python3 ZeroEA_use_tool.py 80 ../data/DBP15K/zh_en/text_input_no_train_11_wxt_KI.txt ../data/DBP15K/zh_en/text_input_no_train_22_wxt_KI.txt ../data/DBP15K/zh_en/text_input_no_train_11_wxt.txt ../data/DBP15K/zh_en/text_input_no_train_22_wxt.txt
```
where the first parameter is the Rouge score threshold; the next two parameters are the input prompts (with tool use) files directory; the fourth and fifth parameters are the input prompts (without tool use) files directory.

Or you can run the following line:
**`bash run/run.sh`**

**And to run all ablation studies**, please go to the folders named "ablation_*" and run the code accordingly.


## Future Work
In this study, we are the first to explore the potential advantages
of incorporating Entity Alignment (EA) techniques to enhance
the performance of downstream tasks. A variety of tasks, such as
dialogue system and vision-based Question Answering (QA), could
significantly benefit from accurate entity alignment. For instance,
vision-based QA necessitates the recognition of Regions of Interest
(ROI) to enable the models to understand objects within the graph.
Thus, we aim to explore more complex downstream tasks, which
consider EA as the intermediate task, thereby initiating a novel
discussion on the influence of EA in complex and realistic scenarios.




