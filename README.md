# Dynamic-Graphs-Construction
Official Codes for Understanding the Dynamics of DNNs Using Graph Modularity (ECCV2022).

*Abstract:* There are good arguments to support the claim that deep neural networks (DNNs) capture better feature representations than the previous hand-crafted feature engineering, which leads to a significant performance improvement. In this paper, we move a tiny step towards understanding the dynamics of feature representations over layers. Specifically, we model the process of class separation of intermediate representations in pre-trained DNNs as the evolution of communities in dynamic graphs. Then, we introduce modularity, a generic metric in graph theory, to quantify the evolution of communities. In the preliminary experiment, we find that modularity roughly tends to increase as the layer goes deeper and the degradation and plateau arise when the model complexity is great relative to the dataset. Through an asymptotic analysis, we prove that modularity can be broadly used for different applications. For example, modularity provides new insights to quantify the difference between feature representations. More crucially, we demonstrate that the degradation and plateau in modularity curves represent redundant layers in DNNs and can be pruned with minimal impact on performance, which provides theoretical guidance for layer pruning. 


We give an example to understand the dynamic graphs:
we leverage pretrained ResNet18 (Top1 acc: 94.78%) on CIFAR-10 to extract feature representations. Then we set N = 50, k = 3 for dynamic graphs construction. The result is shown in the gif below:
![image](https://github.com/yaolu-zjut/Dynamic-Graphs-Construction/blob/main/Dynamic%20Graph%20Construction/gif/cifar10_cResNet18_undirecetd_weighted_network.gif)
Note that nodes of the same color represent samples of the same class. The darker the color of the edge, the higher the similarity between the two corresponding nodes.
The modularity curve of this dynamic graph:
![image](https://github.com/yaolu-zjut/Dynamic-Graphs-Construction/blob/main/Dynamic%20Graph%20Construction/gif/Modularity.jpg)
