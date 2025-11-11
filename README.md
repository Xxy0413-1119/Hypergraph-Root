# Hypergraph-Root
A computational model for the prediction of root-associated proteins

# 1. Description
In Hypergraph-Root, three feature types of proteins are adopted, which are derived from protein sequences. These features are processed by a hypergraph convolutional network and multi-head attention module. Then, the improved features are fed into the fully connected layer for make predictions. 

![Image text](https://github.com/Xxy0413-1119/Hypergraph-Root/blob/main/images/bd1e0c9285b0f74bbc0d32ed7b59c80.png)

# 2. Requirements
## Software Dependencies
pandas = 2.2.3

Python >= 3.10.15

scikit-learn = 1.2.2

pytorch = 2.0.0

pytorch-cuda = 11.8

numpy = 1.26.4

biopython = 1.84

tqdm = 4.66.6

InterPro 104.0 

## Hardware Configuration
Operating System: Windows 10 (64-bit)

CPU: AMD Ryzen 9 7950X

GPU: NVIDIA GeForce RTX 4080

Memory (RAM): 192 GB

CUDA Version: 11.8

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

# 4. Get result
  For each fold in a single cv, you can get the best epoch of the train in `train_result.txt`.
  After all fold trained in a single cv, you can get the evaluation of all fold in `predict_result.txt` and the result of prediction in fold `./result`.
  If you run on `out` mode, there will be only 1 result in `train_result.txt` and `predict_result.txt`.
  If the value of `run` is bigger than 1, former result in `train_result.txt` of a single `cv` or `out` will be override. If you want to save the result of this file, please modify the code on your own.
  Samely, if you run the main.py again, all the result will be override.
