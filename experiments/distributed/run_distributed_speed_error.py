
import pandas as pd
import numpy as np
import torch
import argparse
import pkbar

from pathlib import Path

##############################################################################
##############################################################################

import sys
sys.path.append('./compression')

from rotated_compression import hadamard_compress, hadamard_decompress
from rotated_compression import kashin_compress, kashin_decompress
from rotated_compression import drive_compress, drive_decompress
from rotated_compression import drive_plus_compress, drive_plus_decompress

from rotated_compression import RandomRotation
from rotated_compression import drive_urr_compress, drive_urr_decompress
from rotated_compression import drive_urr_plus_compress, drive_urr_plus_decompress
   
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################

def to_cpu(res):
    return float(res.clone().detach().cpu().numpy())

def compute_error(true_vec, reconstructed_vec):
    return to_cpu(torch.norm(true_vec - reconstructed_vec)/torch.norm(true_vec))**2

##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################

def speed_cost_test(algorithm, dimension, nclients, ntrials, device='cpu', seed=42, verbose=False):
    
    ### matching client-server PRNGs     
    sender_generator = torch.Generator(device=device)
    receiv_generator = torch.Generator(device=device)
    
    sender_generator.manual_seed(seed)
    receiv_generator.manual_seed(seed)
    
    ### collect results
    encode_times = []
    errors = []
    
    print("\n*** Running {} with dimension of {} and {} clients".format(algorithm, dimension, nclients))
    if args.verbose:
        print('\n')

    if args.verbose:
        kbar = pkbar.Kbar(target=ntrials, width=50, always_stateful=True)

    for trial in range(ntrials):
                
        if args.verbose:
            kbar.update(trial)
                
        original_vec = vec_distribution.sample([dimension]).to(device).view(-1)            
        reconstructed_vec = torch.zeros(dimension, device=device)
                    
        for client in range(nclients):

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
                    
            vec = original_vec.detach().clone()

            if algorithm == 'hadamard': 
                start.record()
                cvec, scale = hadamard_compress(vec, dimension, prng=sender_generator)
                end.record()
                reconstructed_vec += hadamard_decompress(cvec, dimension, scale, prng=receiv_generator) 
                
            elif algorithm == 'kashin':
                start.record()
                cvec, scale = kashin_compress(vec, dimension, prng=sender_generator)
                end.record()
                reconstructed_vec += kashin_decompress(cvec, dimension, scale, prng=receiv_generator) 
                
            elif algorithm == 'drive':
                start.record()
                cvec, scale = drive_compress(vec, dimension, prng=sender_generator)
                end.record()
                reconstructed_vec += drive_decompress(cvec, dimension, scale, prng=receiv_generator) 
                
            elif algorithm == 'drive_plus':
                start.record()
                cvec, scale = drive_plus_compress(vec, dimension, prng=sender_generator)
                end.record()
                reconstructed_vec += drive_plus_decompress(cvec, dimension, scale, prng=receiv_generator)
                
            elif algorithm == 'drive_urr': 
                start.record()
                urr = RandomRotation(dimension, vec.device) 
                cvec, scale = drive_urr_compress(vec, dimension, urr)
                end.record()
                reconstructed_vec += drive_urr_decompress(cvec, dimension, scale, urr) 
                
            elif algorithm == 'drive_urr_plus': 
                start.record()
                urr = RandomRotation(dimension, vec.device) 
                cvec, scale = drive_urr_plus_compress(vec, dimension, urr)
                end.record()
                reconstructed_vec += drive_urr_plus_decompress(cvec, dimension, scale, urr)
                               
            else:
                raise Exception("unknown algorithm")
            
            ### torch.cuda.empty_cache()
            
            torch.cuda.synchronize()
            encode_times.append(start.elapsed_time(end))
            
        error = compute_error(original_vec, reconstructed_vec/nclients)
        errors.append(error)
     
    return np.mean(encode_times), np.std(encode_times, ddof=1), np.mean(errors), np.std(errors, ddof=1)    

##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################

def encode_speed(algorithm, dimension, nvectors, ntrials, device='cpu', seed=42, verbose=False):
        
    sender_generator = torch.Generator(device=device)
    sender_generator.manual_seed(seed)
            
    encode_times   = []

    print("\n*** Running {} with dimension of {} and {} repetitions".format(algorithm, dimension, ntrials))
    if args.verbose:
        kbar = pkbar.Kbar(target=nvectors, width=50, always_stateful=True)
        print('\n')
        
    for vector in range(nvectors):

        if args.verbose:
            kbar.update(vector)
                    
        original_vec = vec_distribution.sample([dimension]).to(device).view(-1) 

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
            
        if algorithm == 'hadamard': 
            start.record()
            for i in range(ntrials):
                vec = original_vec.clone()
                cvec, scale = hadamard_compress(vec, dimension, prng=sender_generator)
            end.record()
                        
        elif algorithm == 'kashin':
            start.record()
            for i in range(ntrials):
                vec = original_vec.clone()
                cvec, scale = kashin_compress(vec, dimension, prng=sender_generator)
            end.record()
            
        elif algorithm == 'drive':
            start.record()
            for i in range(ntrials):
                vec = original_vec.clone()
                cvec, scale = drive_compress(vec, dimension, prng=sender_generator)
            end.record()
            
        elif algorithm == 'drive_plus':
            start.record()
            for i in range(ntrials):
                vec = original_vec.clone()
                cvec, scale = drive_plus_compress(vec, dimension, prng=sender_generator)
            end.record()
            
        elif algorithm == 'drive_urr': 
            start.record()
            for i in range(ntrials):
                vec = original_vec.clone()
                urr = RandomRotation(dimension, vec.device) 
                cvec, scale = drive_urr_compress(vec, dimension, urr)
            end.record()
            
        elif algorithm == 'drive_urr_plus': 
            start.record()
            for i in range(ntrials):
                vec = original_vec.clone()
                urr = RandomRotation(dimension, vec.device) 
                cvec, scale = drive_urr_plus_compress(vec, dimension, urr)
            end.record()
                           
        else:
            raise Exception("unknown algorithm")
        
        ### torch.cuda.empty_cache()
        
        torch.cuda.synchronize()
        encode_times.append(start.elapsed_time(end) / ntrials)   
        
    return np.mean(encode_times), np.std(encode_times, ddof=1)
                
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################

def figure_tests(args):

    ### configuration
    nclients = 10
    
    ### dimension to scan
    dimensions = [2**i for i in range(2,12)]
            
    ### create directory if needed
    Path(args.path).mkdir(parents=True, exist_ok=True)
    Path(args.path + "/figure_results").mkdir(parents=True, exist_ok=True)
    
    ### algorithms
    algorithms = ['drive', 'drive_plus', 'drive_urr', 'drive_urr_plus']
    
    ### table
    df = pd.DataFrame(columns=algorithms)  
    for dim in dimensions:
        df = df.append(pd.Series(name=dim))      
    
    ### hadamard
    algorithms = ['drive', 'drive_plus']
    nclients_ntrials = 10000
           
    for dim in dimensions:
        for alg in algorithms:
            _, _, err, err_std = speed_cost_test(alg, dim, nclients, nclients_ntrials, device=device, seed=args.seed, verbose=args.verbose)
            df.loc[dim].at[alg] = "{:.4f}, {:.4f}".format(err, err_std)
              
    ### uniform random rotation
    algorithms = ['drive_urr', 'drive_urr_plus']
    nclients_ntrials = 100
    
    for dim in dimensions:
        for alg in algorithms:
            _, _, err, err_std = speed_cost_test(alg, dim, nclients, nclients_ntrials, device=device, seed=args.seed, verbose=args.verbose)
            df.loc[dim].at[alg] = "{:.4f}, {:.4f}".format(err, err_std)

    df.to_pickle(args.path + "/figure_results/cost_{}.pkl".format(str(type(vec_distribution)).split(".")[-2]))
           

##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################

def table_tests(args):
    
    algorithms = ['hadamard',
                  'kashin', 
                  'drive', 
                  'drive_plus',
                  'drive_urr', 
                  'drive_urr_plus'
                  ]
   
    dimensions = [128, 8192, 524288, 33554432]
    
    nclients = 10
    nclients_ntrials = {
            128 : 1000,
            8192 : 1000,
            524288 : 1000,
            33554432 : 100
    }
        
    nvectors = 100
    nvectors_ntrials = {
            128 : 100,
            8192 : 100,
            524288 : 100,
            33554432 : 10
    }
    
    ### create directory if needed
    Path(args.path).mkdir(parents=True, exist_ok=True)
    Path(args.path + "/table_results").mkdir(parents=True, exist_ok=True)
       
    ### test 1
    df1 = pd.DataFrame(columns=algorithms)  
    for dim in dimensions:
        df1 = df1.append(pd.Series(name=dim))
        
    for dim in dimensions:
        for alg in algorithms:

            ### limit on uniform random rotation
            if 'urr' in alg and dim > 2**13:
                continue
            
            ent, _, err, _ = speed_cost_test(alg, dim, nclients, nclients_ntrials[dim], device=device, seed=args.seed, verbose=args.verbose)
            df1.loc[dim].at[alg] = "{:.4f}, {:.4f}".format(err, ent)

    df1.to_pickle(args.path + "/table_results/cost_speed_{}.pkl".format(str(type(vec_distribution)).split(".")[-2]))
           
    ### test 2
    df2 = pd.DataFrame(columns=algorithms)  
    for dim in dimensions:
        df2 = df2.append(pd.Series(name=dim))
    
    for dim in dimensions:
        for alg in algorithms:

            ### limit on uniform random rotation
            if 'urr' in alg and dim > 2**13:
                continue
            
            ent, _ = encode_speed(alg, dim, nvectors, nvectors_ntrials[dim], device=device, seed=args.seed, verbose=args.verbose)
            df2.loc[dim].at[alg] = "{:.4f}".format(ent)

    df2.to_pickle(args.path + "/table_results/encode_speed_{}.pkl".format(str(type(vec_distribution)).split(".")[-2]))
    
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
           
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="""
        Generates the data for table 1 and figure 1 in the paper.
        The results are saved as a pandas dataframe (rows are dimensions / columns are algorithm names).
        The results folder is: <path> / {figure_results}/{table_results} / {name}.pkl""",
                                     formatter_class=argparse.RawTextHelpFormatter)
       
    ### verbosity
    parser.add_argument('--verbose', default=True, action='store_true', help='detailed progress')

    ### GPU index to work with, if more than one is available.
    parser.add_argument('--gpu', default='0', type=str, help='gpu index to run the simulation')

    ### seed
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    
    ### path to result folder
    parser.add_argument('--path', default='./results/distributed_speed_error', type=str, help='random seed')    

    ### dataset
    parser.add_argument('--test', default='figure1', choices=['figure1', 'table1'], help='which data to generate')

    ### distribution
    parser.add_argument('--dist', default='all', choices=['lognormal', 'normal', 'exponential', 'all'], help='which distributions to run')
    
    args = parser.parse_args()

    ##########################################################################
    ##########################################################################

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda' and torch.cuda.device_count() > 1:
        device = device + ':{}'.format(args.gpu)
    print("running on device: {}".format(device))
          
    torch.manual_seed(args.seed)
    
    ##########################################################################
    ##########################################################################

    if args.dist == 'lognormal':
        vec_distributions = [torch.distributions.LogNormal(0,1)]
    elif args.dist == 'normal':
        vec_distributions = [torch.distributions.Normal(0,1)]
    elif args.dist == 'exp':
        vec_distributions = [torch.distributions.exponential.Exponential(torch.Tensor([1]))]
    elif args.dist == 'all':
        vec_distributions = [torch.distributions.LogNormal(0,1), 
                             torch.distributions.Normal(0,1), 
                             torch.distributions.exponential.Exponential(torch.Tensor([1.0]))]
    else:
        raise Exception("unsupported distribution")
        
    ##########################################################################
    ##########################################################################
                
    print("Generating data for {}.".format(args.test))
    
    for vec_distribution in vec_distributions:
    
        if args.test == 'table1':
            table_tests(args) 
        elif args.test == 'figure1':   
            figure_tests(args) 
        else:
            raise Exception("unknown test type")
        
    ##########################################################################
    ##########################################################################