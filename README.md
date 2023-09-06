# ZeroEA: A Zero-Training Entity Alignment Framework via Pre-Trained Language Model

[![Python 3.7+](https://img.shields.io/badge/Python-3.7+-teal.svg)](https://www.python.org/downloads/release/python-390/)
[![Pytorch 1.8+](https://img.shields.io/badge/Pytorch-1.8+-red.svg)](https://pytorch.org/blog/pytorch-1.8-released/)
[![Transformers 4.8+](https://img.shields.io/badge/Transformers-4.8+-cine.svg)](https://pypi.org/project/transformers/)
[![BERT 1.8+](https://img.shields.io/badge/BERT-PLM-blue.svg)](https://huggingface.co/bert-base-uncased)
[![motif 1.8+](https://img.shields.io/badge/Motif-graph-yellow.svg)](https://en.wikipedia.org/wiki/Network_motif)
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



### Run Experiments

**To run ZeroEA please use**

**`bash run/run.sh`**

**And to run all ablation studies**, please go to the folders named "ablation_*" and run the code accordingly.
