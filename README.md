# DRIVE: One-bit Distributed Mean Estimation

This repository is the official implementation of 'DRIVE: One-bit Distributed Mean Estimation'.


## Context

*DRIVE* is a 1-bit compression algorithm for distributed mean estimation. When applied to various distributed and specifically federated learning tasks, it shows consistent improvement over the state of the art.

## Folder structure 

`drive_tf` and `drive_torch` folders contain our standalone implementation for TensorFlow and PyTorch, respectively.

The `experiments` folder contains details on how to reproduce the paper's results. It is separated into two sub-projects: 
1) The `distributed` sub-folder contains all the distributed learning experiments (Distributed CNN, K-means, Power Iteration) using PyTorch. 
2) The `federated` sub-folder contains the federated learning experiments using TensorFlow.  
