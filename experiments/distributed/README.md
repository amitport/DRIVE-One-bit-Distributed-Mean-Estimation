# Drive: One-bit Distributed Mean Estimation

This repository contains all the distributed tasks (Distributed CNN, K-means, Power Iteration) presented in the evaluation. 

## Setup:
- To install all the dependencies, run: `pip3 install -r requirments.txt`
- To execute the count sketch simulations, import csh by running *in (this) main project folder*: `git clone https://github.com/nikitaivkin/csh`

## Files:
- distributed_cnn.py: 

  Simulation of distributed CNN with different compression algorithms.<br/>
  The results can be found in the results/distributed_cnn folder.<br/>
  The results filename is: results_<dataset>_<compression_alg>_<clients>_<lr (without ".")>.pkl<br/>
  
  For more information, run: `python3 distributed_cnn.py -h`
  
- distributed_kmeans.py:
  
  Simulation of distributed K-means with different compression algorithms.<br/>
  The results can be found in the results/distributed_kmeans folder.<br/>
  The results filename is: results_<dataset>_<compression_alg>_<clients>.pkl<br/>
  
  For more information, run: `python3 distributed_kmeans.py -h` 
  
- distributed_power_iteration.py:
  
  Simulation of distributed power iteration with different compression algorithms.<br/>
  The results can be found in the results/distributed_power_iteration folder.<br/>
  The results filename is: results_<dataset>_<compression_alg>_<clients>_<lr (without ".")>.pkl<br/>
  
  For more information, run: `python3 distributed_power_iteration.py -h` 

- compression/compression.py and compression/rotated_compression.py:
  
  Contain the compression and decompression implementations for all of the evaluated algorithms in the paper. 

## How to run the simulations?
- `python3 run_distributed_cnn_sims.py`:
  
  Executes all of the distributed CNN training tasks with 10 clients 
  in two configurations: (1) CIFAR-10 dataset with ResNet-9; (2) CIFAR-100 with ResNet-18.<br/>
  The simulations are executed with different comporession alghorithms: FedAvg, TernGrad, Sketched-SGD, Hadamard + 1-bit SQ, Kashin + 1-bit SQ, Drive (Hadamard), Drive+ (Hadamard).<br/>
  
  The result files can be found in the results/distributed-cnn folder.

- `python3 run_distributed_kmeans_sims.py`:
  
  Executes all of the distributed K-means tasks with 10, 100 and 1000 clients and two dataset: MNIST and CIFAR-10.<br/>
  The simulations are executed with different comporession alghorithms: Baseline, Hadamard + 1-bit SQ, Kashin + 1-bit SQ, Drive (Hadamard), Drive+ (Hadamard).<br/>

  The results can be found in the results/distributed_kmeans folder.

- `python3 run_distributed_power_iteration_sims.py`:

  Executes all of the distributed Power Iteration tasks with 10, 100 and 1000 clients and two dataset: MNIST and CIFAR-10.<br/>
  The simulations are executed with different comporession alghorithms: Baseline, Hadamard + 1-bit SQ, Kashin + 1-bit SQ, Drive (Hadamard), Drive+ (Hadamard).<br/>

  The results can be found in the results/distributed_power_iteration folder.

  
- `python3 run_distributed_speed_error.py`:

  This file includes two tests (see the results in Figure 1 and Table 1 in the paper) that measure the encoding speed and NMSE of: Hadamard + 1-bit SQ, Kashin + 1-bit SQ, Drive (Hadamard), Drive+ (Hadamard), Drive (Uniform), Drive+ (Uniform).<br/>
  For both tests (set the desired test by the --test flag to either 'table1' or 'figure1'), the results are saved to a pickle file as pandas dataframes (each row specifies the dimension; each column specifies the algorithm).<br/> 
  
## How to plot the results? 

In order to plot the results, run: `python3 results/plot.py --plot <graph_name>`<br/>
  where <*graph_name*> options are: <br/>
  *distributed-cnn* (to generate Figure 3)<br/>
  *distributed-power-iteration* (to generate Figure 4)<br/>
  *distributed-kmeans* (to generate Figure 5)<br/>
  *distributed-algs-appendix* (to generate Figure 6)<br/>

  
  
  
  
