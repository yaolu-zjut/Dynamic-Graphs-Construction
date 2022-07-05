# Dynamic-Graphs-Construction
Official Codes for Understanding the Dynamics of DNNs Using Graph Modularity(ECCV2022).

We give an example to understand the dynamic graphs:
we leverage pretrained ResNet18 (Top1 acc: 94.78%) on CIFAR-10 to extract feature representations. Then we set N = 50, k = 3 for dynamic graphs construction. The result is shown in the gif below:
![image](https://github.com/yaolu-zjut/Dynamic-Graphs-Construction/blob/main/Dynamic%20Graph%20Construction/gif/cifar10_cResNet18_undirecetd_weighted_network.gif)
Note that nodes of the same color represent samples of the same class. The darker the color of the edge, the higher the similarity between the two corresponding nodes.
The modularity curve of this dynamic graph:
![image](https://github.com/yaolu-zjut/Dynamic-Graphs-Construction/blob/main/Dynamic%20Graph%20Construction/gif/Modularity.jpg)
