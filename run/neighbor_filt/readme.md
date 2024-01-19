# Count and Enumerate Motifs in ZeroEA (VLDB 2024 under review)
Nan Huo, Reynold Cheng, Ben Kao, Wentao Ning, Nur Al Hasan Haldar, Xiaodong Li, Jinyang Li, Tian Li, Mohammad Matin Najafi, Ge Qu  

<p align="center" width="100%">
<a><img src="../../img/ZeroEA.png" style="width: 100%; min-width: 300px; display: block; margin: auto;"></a>
</p>

## Codes Usage
ZeroEA is a novel **zero-training** entity alignment framework for knowledge graphs. It bridges the gap between graph structure and plain text by converting KG topology into textual context suitable for PLM input. Additionally, in order to provide PLMs with concise and clear input text of reasonable length, we design a motif-based neighborhood filter. It includes an offline global motif counting module [1] and an online local motif enumeration module [2]. The two modules are with the same data graph format, i.e., each line as an edge seperated by a blank space. 

## Global Motif Counting
For the offline global motif counting module, we employ ESCAPE [1], the state-of-the-art algorithm in motif counting domain. It will return the frequency of motifs whose sizes are smaller than 5, and we need to make sure that the instances of the motif are abundant, and this motif is commonly used in the correnponding area or domain.

```bash
git clone https://bitbucket.org/seshadhri/escape/src/master/
cd master
make
cd python
python sanitize.py ../../ graphs datagraph
cd ../wrapper
python3 subgraph_counts.py ../../graphs/datagraph.edges 5 -i
```

## Local Motif Enumeration

For the online local motif enumeration module, we implememnt E-CLog, the state-of-the-art algorithm in local motif enumeration domain. It will return the one-hop neighbors that are within the same motif instace as the query node. 
```bash
javac *.java
java mfilter ./ graphs/datagraph triangle 10
```


## Reference
[1] Dave, V. S., Ahmed, N. K., & Al Hasan, M. (2017, December). E-CLoG: counting edge-centric local graphlets. In 2017 IEEE International Conference on Big Data (Big Data) (pp. 586-595). IEEE.  
[2] Pinar, A., Seshadhri, C., & Vishal, V. (2017, April). Escape: Efficiently counting all 5-vertex subgraphs. In Proceedings of the 26th international conference on world wide web (pp. 1431-1440).
