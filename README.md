# DRIVE: One-bit Distributed Mean Estimation

This repository is the official implementation of the paper 'DRIVE: One-bit Distributed Mean Estimation', which was published at [NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/0397758f8990c1b41b81b43ac389ab9f-Abstract.html).


## Context

*DRIVE* is a 1-bit compression algorithm for distributed mean estimation. When applied to various distributed and specifically federated learning tasks, it shows consistent improvement over the state of the art.

## Folder structure 

`drive_tf` and `drive_torch` folders contain our standalone implementation for TensorFlow and PyTorch, respectively.

The `experiments` folder contains details on how to reproduce the paper's results. It is separated into two sub-projects: 
1) The `distributed` sub-folder contains all the distributed learning experiments (Distributed CNN, K-means, Power Iteration) using PyTorch. 
2) The `federated` sub-folder contains the federated learning experiments using TensorFlow.  

## Citation

If you find this useful, please cite us:

```bibtex
@inproceedings{NEURIPS2021_0397758f,
 author = {Vargaftik, Shay and Ben-Basat, Ran and Portnoy, Amit and Mendelson, Gal and Ben-Itzhak, Yaniv and Mitzenmacher, Michael},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {M. Ranzato and A. Beygelzimer and Y. Dauphin and P.S. Liang and J. Wortman Vaughan},
 pages = {362--377},
 publisher = {Curran Associates, Inc.},
 title = {DRIVE: One-bit Distributed Mean Estimation},
 url = {https://proceedings.neurips.cc/paper/2021/file/0397758f8990c1b41b81b43ac389ab9f-Paper.pdf},
 volume = {34},
 year = {2021}
}
```
