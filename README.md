# DRIVE: One-bit Distributed Mean Estimation

This repository is the official implementation of [DRIVE: One-bit Distributed Mean Estimation](https://arxiv.org/abs/2105.08339).


## Context

*DRIVE* is a 1-bit compression algorithm for distributed mean estimation. When applied to various distributed and specifically federated learning tasks, it shows consistent improvement over the state of the art.

## Folder structure 

`drive_tf` and `drive_torch` folders contain our implementation for TensorFlow and PyTorch, respectively.

The `experiments` folder contains details on how to reproduce the paper's results.

## License

The code is released under the MIT license. 

If you find this useful, please cite our work: 


```bibtex
@misc{vargaftik2021drive,
      title={DRIVE: One-bit Distributed Mean Estimation}, 
      author={Shay Vargaftik and Ran Ben Basat and Amit Portnoy and Gal Mendelson and Yaniv Ben-Itzhak and Michael Mitzenmacher},
      year={2021},
      eprint={2105.08339},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
