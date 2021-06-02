
import random
import numpy as np
import argparse
import os
import dataset_factories
import pickle
import torch
import pkbar

from multiprocessing import freeze_support
from defs import get_device

import sys
sys.path.append('./compression')

from compression import fedavg_compress, fedavg_decompress
from rotated_compression import hadamard_compress, hadamard_decompress
from rotated_compression import kashin_compress, kashin_decompress
from rotated_compression import drive_compress, drive_decompress
from rotated_compression import drive_plus_compress, drive_plus_decompress

##############################################################################
##############################################################################

def get_suffix(args):
    return '{}_{}_{}_{}'.format(args.dataset, args.compression_alg, args.clients, str(args.lr).replace('.', ''))

##############################################################################
##############################################################################

def iteration_h(iter_args):

        dataloader = iter_args['dataloader']
        dataset_len = iter_args['dataset_len']
        batch_size = iter_args['batch_size']

        client_sums = []
        for client, (client_inputs, client_targets) in enumerate(dataloader):
            
            ### every client has the same batch size 
            if client_inputs.shape[0] != batch_size:
                continue
            
            A = np.array([client_inputs[i].view(-1).numpy() for i in range(batch_size)])
            client_sums.append(np.sum(A, axis=0))
            
        meanvector = np.sum(client_sums, axis=0) / dataset_len

        return meanvector

def iteration(iter_args):
        
        dataloader = iter_args['dataloader']
        batch_size = iter_args['batch_size']

        eigenvector = iter_args['eigenvector']
        meanvector = iter_args['meanvector']

        epoch = iter_args['epoch']

        ### progress bar
        if args.verbose:
            kbar = pkbar.Kbar(target=len(dataloader), epoch=epoch-1, num_epochs=args.num_epochs, width=50, always_stateful=True)
        
        compressed_client_vectors = {}

        for client, (client_inputs, client_targets) in enumerate(dataloader):

            ### every client has the same batch size
            if client_inputs.shape[0] != batch_size:
                continue

            A = np.array([client_inputs[i].view(-1).numpy() for i in range(batch_size)])
            A = A - meanvector

            s = np.zeros(A.shape[1])

            for x in A:
                s = s + np.dot(x, eigenvector) * x

            ### compute, compress and send parameter diffs
            client_diff_vector = torch.Tensor(s - iter_args['client_states'][client]).to(device)

            if args.compression_alg in rotated_algorithms:
                if need_padding:
                    padded_client_vec = torch.zeros(padded_dimension, device=device)
                    padded_client_vec[:dim] = client_diff_vector
                    compressed_client_vectors[client] = compress(padded_client_vec, padded_dimension, prng=snd_prngs[client])
                else:
                    compressed_client_vectors[client] = compress(client_diff_vector, padded_dimension, prng=snd_prngs[client])
            else:
                compressed_client_vectors[client] = compress(client_diff_vector, params=params, prng=snd_prngs[client])

            ##############################################################
            ############## print status ##################################
            ##############################################################
            if args.verbose:
                kbar.update(client+1)

        for compressed_client_index, compressed_client_vector in compressed_client_vectors.items():
            ccv, ccv_metadata = compressed_client_vector

            ### update local status ("what the parameter server thinks")
            if args.compression_alg in rotated_algorithms:
                iter_args['client_states'][compressed_client_index] += args.lr * decompress(ccv, padded_dimension,
                                    ccv_metadata, prng=rcv_prngs[compressed_client_index]).cpu().numpy()[:dim]
            else:
                iter_args['client_states'][compressed_client_index] += args.lr * decompress(ccv, ccv_metadata, params,
                                                            prng=rcv_prngs[compressed_client_index]).cpu().numpy()

        ##################################################################
        ################# compute the averaged eigenvector ###############
        ##################################################################

        new_eigenvector = sum(iter_args['client_states'].values())

        eigenvalue = np.dot(eigenvector, new_eigenvector)

        l2_error = np.linalg.norm(eigenvalue * eigenvector - new_eigenvector)

        new_eigenvector = new_eigenvector / np.linalg.norm(new_eigenvector)

        return new_eigenvector, l2_error

##############################################################################
##############################################################################

if __name__ == '__main__':

    freeze_support()

    parser = argparse.ArgumentParser(description="""
        Simulation of distributed power iteration with different compression algorithms.
        The results can be found in the results/distributed_power_iteration folder.
        The results filename is: results_<dataset>_<compression_alg>_<clients>_<lr (without ".")>.pkl""",
        formatter_class=argparse.RawTextHelpFormatter)

    ### training epochs
    parser.add_argument('--num_epochs', default=200, type=int, help='Number of epochs to run')

    ### number of clients
    parser.add_argument('--clients', default=100, type=int, help='Number of clients')

    ### dataset
    parser.add_argument('--dataset', default='MNIST_DA', choices=['MNIST_DA', 'CIFAR10_DA'], help='')

    ### compression scheme
    parser.add_argument('--compression_alg', default='baseline',
                        choices=['baseline', 'hadamard', 'kashin', 'drive', 'drive_plus'],
                        help='compression/decompression scheme')

    ### learning rate
    parser.add_argument('--lr', default=0.1, type=float, help='Learning rate')
    
    ### print info to screen
    parser.add_argument('--verbose', default=False, action="store_true", help='Print more info')

    ### GPU index to work with, if more than one is available.
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU index to run the simulation (if more than one is available)')

    ### seed
    parser.add_argument('--seed', default=42, type=int, help='Seed to prngs')

    args = parser.parse_args()

    # print simulation info
    print ('*** Running distributed power iteration with {} over {} clients with {} dataset with learning rate '
           'of {} over {} epochs'.format(args.compression_alg, args.clients, args.dataset, args.lr, args.num_epochs))

    device, _ = get_device(args.gpu)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    ##########################################################################
    ####################### Preparing data ###################################
    ##########################################################################

    print('==> Preparing data, this might take a while...')

    dataloader, dataset_features, dataset_labels, dataset_len, num_classes, batch_size, dim = \
        getattr(dataset_factories, args.dataset)(args.clients)

    params = {}
    params['device'] = device[0]

    padded_dimension = dim
    if not dim & (dim - 1) == 0:
        padded_dimension = int(2 ** (np.ceil(np.log2(dim))))
    need_padding = padded_dimension != dim

    rotated_algorithms = ['hadamard', 'kashin', 'drive', 'drive_plus']

    ##########################################################################
    ########################## PRNGs #########################################
    ##########################################################################

    rcv_prngs = {}
    snd_prngs = {}

    for client_index in range(args.clients):
        seed = np.random.randint(2 ** 31)

        rgen = torch.Generator(device=device)
        rgen.manual_seed(seed)

        rcv_prngs[client_index] = rgen

        sgen = torch.Generator(device=device)
        sgen.manual_seed(seed)

        snd_prngs[client_index] = sgen

    ##########################################################################
    ####################### Compression Settings #############################
    ##########################################################################

    if args.compression_alg == 'baseline':

        compress = fedavg_compress
        decompress = fedavg_decompress

    ##########################################################################

    elif args.compression_alg == 'hadamard':

        compress = hadamard_compress
        decompress = hadamard_decompress

    ##########################################################################

    elif args.compression_alg == 'kashin':

        compress = kashin_compress
        decompress = kashin_decompress

    ##########################################################################

    elif args.compression_alg == 'drive':

        compress = drive_compress
        decompress = drive_decompress

    ##########################################################################

    elif args.compression_alg == 'drive_plus':

        compress = drive_plus_compress
        decompress = drive_plus_decompress

    else:

        raise Exception('compression algorithm undefined. Received {}'.format(args.compression_alg))

    ##########################################################################
    ####################### Run ##############################################
    ##########################################################################

    if not os.path.isdir('results'):
        os.mkdir('results')

    if not os.path.isdir('results/distributed_power_iteration'):
        os.mkdir('results/distributed_power_iteration')

    suffix = get_suffix(args)

    # collect stats to lists
    results = {
        'epochs': [],
        'L2_error': [],
    }

    ##########################################################################

    eigenvector = np.random.rand(dim)
    eigenvector = eigenvector / np.linalg.norm(eigenvector)

    iter_args = {}

    iter_args['dataloader'] = dataloader
    iter_args['batch_size'] = batch_size
    iter_args['dataset_len'] = dataset_len

    iter_args['meanvector'] = iteration_h(iter_args)

    iter_args['client_states'] = {}
    for i in range(args.clients):
        iter_args['client_states'][i] = np.zeros(dim)

    if not args.verbose:
        kbar = pkbar.Kbar(target=args.num_epochs, width=50, always_stateful=True)

    for epoch in range(args.num_epochs):

        iter_args['epoch'] = epoch + 1
        iter_args['eigenvector'] = eigenvector

        eigenvector, error = iteration(iter_args)

        results['epochs'].append(epoch)
        results['L2_error'].append(error)

        if not args.verbose:
            kbar.update(epoch+1, values=[("L2 error", error)])

        if error > 1e16:
            break

    with open('./results/distributed_power_iteration/' + 'results_' + suffix + '.pkl', 'wb') as filehandle:
        pickle.dump(results, filehandle)
