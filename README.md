# Hypergraph--Root
Root-Associated Proteins in Plants Prediction Model Based on Hypergraph Convolutional Network

# 1. Description
In the Hypergraph-Root model, the features of proteins are extracted based on the raw features of their amino acids.These raw features are first fed into the Hypergraph Convolutional Network for initial processing to fully exploit the complex relationships and potential information between amino acids.Subsequently, the features processed by the hypergraph convolutional network will be fed into the multi-head attention module, which is able to deeply analyze and weight and integrate the features from different perspectives to further enhance the expressive ability and differentiation of the features.Finally, the features that have been synergistically optimized by the hypergraph convolutional network and the multi-head attention module will be sent to the fully connected layer, which generates the prediction results of protein classes through a series of complex linear transformations and nonlinear activation operations.

# 2. Requirements
pandas = 2.2.3

Python >= 3.10.15

scikit-learn = 1.2.2

pytorch = 2.0.0

pytorch-cuda = 11.8

numpy = 1.26.4

biopython = 1.84

tqdm = 4.66.6

# 3. How to use
1.Set up your enviroment and download the code from github

2.Put your data into the appropriate folder:
  ```
     [Protein contact map] --> ./data/graph    
     [PSSM feature] --> ./data/pssm
     [ProtT5 feature] --> ./data/prot
  ```
3. Activate your enviroment and run main.py:
  ```
     $ python main.py --mode cv --run 10
     or
     $ python main.py --mode out --run 10
  ```
  Within this line of code, you can choose between cross-validation mode or independent test set mode by modifying the value after mode.
  option  | value |
| ------------- | ------------- |
| `mode` | `cv`or`out` |
| `run` | int value for run times |
