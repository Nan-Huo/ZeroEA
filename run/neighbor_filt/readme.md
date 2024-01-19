# Count and Enumerate Motifs in ZeroEA (VLDB 2024 under review)
Nan Huo, Reynold Cheng, Ben Kao, Wentao Ning, Nur Al Hasan Haldar, Xiaodong Li, Jinyang Li, Tian Li, Mohammad Matin Najafi, Ge Qu  

<p align="center" width="100%">
<a><img src="../../img/ZeroEA.png" style="width: 100%; min-width: 300px; display: block; margin: auto;"></a>
</p>

## Global Motif Counting
ZeroEA is a novel **zero-training** entity alignment framework for knowledge graphs. It bridges the gap between graph structure and plain text by converting KG topology into textual context suitable for PLM input. Additionally, in order to provide PLMs with concise and clear input text of reasonable length, we design a motif-based neighborhood
filter to eliminate noisy neighbors. Notably, ZeroEA can **outperform state-of-the-art supervised baselines**, and our study highlights the considerable potential of EA technique in improving the performance of downstream tasks, thereby benefitting the broader research field.


## Local Motif Enumeration

• Please create the virtual environment and activate it through:
```bash
conda create -n zeroea python=3.7
source activate zeroea
```
• And then download the dependencies in **requirements.txt** file through:
```bash
pip install -r requirements.txt
```


## Reference
[1] Dave, V. S., Ahmed, N. K., & Al Hasan, M. (2017, December). E-CLoG: counting edge-centric local graphlets. In 2017 IEEE International Conference on Big Data (Big Data) (pp. 586-595). IEEE.  
[2] Pinar, A., Seshadhri, C., & Vishal, V. (2017, April). Escape: Efficiently counting all 5-vertex subgraphs. In Proceedings of the 26th international conference on world wide web (pp. 1431-1440).
