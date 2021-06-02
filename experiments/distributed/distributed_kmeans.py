
import random
import numpy as np
import os
import argparse
import dataset_factories
import torch
import pickle
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
    return '{}_{}_{}'.format(args.dataset, args.compression_alg, args.clients)

##############################################################################
##############################################################################

def iteration(dataset_args, centers):
    
        ### dataset iterator
        dataloader = dataset_args["dataloader"]

        ### dataset attributes
        num_classes = dataset_args["num_classes"]
        batch_size = dataset_args["batch_size"]
        dim = dataset_args["dim"]

        ### global train objective
        train_objective = 0

        ### client auxilary lists
        center_mass = {}
        compressed_center_sums = {}
        
        ### progress bar
        if args.verbose:
            kbar = pkbar.Kbar(target=len(dataloader), epoch=epoch, num_epochs=args.num_epochs, width=50, always_stateful=True)

        ### iterate over clients
        for client, (client_inputs, client_targets) in enumerate(dataloader):
            
            ### every client has the same batch size 
            if client_inputs.shape[0] != batch_size:
                continue
            
            ### flatten images to vectors
            client_inputs = client_inputs.view(batch_size, dim).to(device)

            ### calculate the L2 distance from each point to each center
            client_input_distances = torch.Tensor(batch_size, num_classes).to(device)
            for i, center in enumerate(centers):
                client_input_distances[:, i] = ((client_inputs - center[None, :]) ** 2).sum(dim=1)

                ### index of the closest center to each input
            client_input_closest_centers = torch.argmin(client_input_distances, dim=1)

            ### calculate current client objective
            train_objective += ((client_inputs.to(device).view(client_inputs.shape[0], dim) - centers[
                client_input_closest_centers]) ** 2).sum()

            ### calculate new client centers and their masses
            client_center_sums = torch.zeros(num_classes, dim, device=device)
            client_center_mass = torch.zeros(num_classes, device=device)

            for i in range(num_classes):
                client_center_sums[i] = client_inputs[client_input_closest_centers == i].sum(0)
                client_center_mass[i] = (client_input_closest_centers == i).sum()

            center_mass[client] = client_center_mass

            if args.compression_alg in rotated_algorithms:
                if need_padding:
                    padded_client_vec = torch.zeros(padded_dimension, device=device)
                    padded_client_vec[:dataset_args["dim"] * dataset_args["num_classes"]] = client_center_sums.view(-1)
                    compressed_center_sums[client] = compress(padded_client_vec, padded_dimension,
                                                              prng=snd_prngs[client])
                else:
                    compressed_center_sums[client] = compress(client_center_sums.view(-1), padded_dimension,
                                                              prng=snd_prngs[client])
            else:
                compressed_center_sums[client] = compress(client_center_sums.view(-1), params=params,
                                                          prng=snd_prngs[client])

            ### print status
            if args.verbose:
                kbar.update(client+1, values=[("L2 error", train_objective / (client + 1))])

        ##################################################################
        ################# decomporess by the server ######################
        ##################################################################

        center_mass_sum = torch.zeros(num_classes, device=device)
        centers_sum = torch.zeros(dim*num_classes, device=device)

        for  client_index, compressed_center_sums_client in compressed_center_sums.items():
            ### sum centers mass
            center_mass_sum += center_mass[client_index]

            ccv, ccv_metadata = compressed_center_sums_client
            if args.compression_alg in rotated_algorithms:
                centers_sum += decompress(ccv, padded_dimension, ccv_metadata,
                                          prng=rcv_prngs[client_index])[:dim*num_classes]
            else:
                centers_sum += decompress(ccv, ccv_metadata, params, prng=rcv_prngs[client_index])

        ### calculate new global centers
        new_centers = centers_sum.view(num_classes, dim) / center_mass_sum[:, None]

        ### nan to 0
        new_centers[new_centers.isnan()] = 0
        
        ### return global objective and the new centers
        return train_objective.cpu().numpy(), new_centers

##############################################################################
##############################################################################

if __name__ == '__main__':

    freeze_support()

    parser = argparse.ArgumentParser(description="""
        Simulation of distributed K-means with different compression algorithms.
        The results can be found in the results/distributed_kmeans folder.
        The results filename is: results_<dataset>_<compression_alg>_<clients>.pkl""",
        formatter_class=argparse.RawTextHelpFormatter)

    ### training epochs
    parser.add_argument('--num_epochs', default=45, type=int, help='Number of epochs to run')

    ### number of clients
    parser.add_argument('--clients', default=100, type=int, help='Number of clients')

    ### dataset
    parser.add_argument('--dataset', default='MNIST_DA', choices=['MNIST_DA', 'CIFAR10_DA'])

    ### compression scheme
    parser.add_argument('--compression_alg', default='baseline',
                        choices=['baseline', 'hadamard', 'kashin', 'drive', 'drive_plus'],
                        help='compression/decompression scheme')

    ### GPU index to work with, if more than one is available.
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU index to run the simulation (if more than one is available)')

    ### seed
    parser.add_argument('--seed', default=42, type=int, help='Seed to prngs')

    ### print info to screen
    parser.add_argument('--verbose', default=False, action="store_true", help='Print more info')

    ### parse args
    args = parser.parse_args()

    # print simulation info
    print ('*** Running distributed K-means with {} over {} clients with {} dataset over {} epochs'.
           format(args.compression_alg, args.clients, args.dataset, args.num_epochs))

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

    dataset_args = {}

    dataset_args["dataloader"] = dataloader
    dataset_args["dataset_features"] = dataset_features
    dataset_args["dataset_labels"] = dataset_labels
    dataset_args["dataset_len"] = dataset_len
    dataset_args["num_classes"] = num_classes
    dataset_args["batch_size"] = batch_size
    dataset_args["dim"] = dim

    params = {}
    params['device'] = device

    rotated_algorithms = ['hadamard', 'kashin', 'drive', 'drive_plus']

    padded_dimension = dim * num_classes

    if not padded_dimension & (padded_dimension-1) == 0:
        padded_dimension = int(2**(np.ceil(np.log2(dim * num_classes))))
    need_padding = padded_dimension != dim * num_classes

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

    if not os.path.isdir('results/distributed_kmeans'):
        os.mkdir('results/distributed_kmeans')

    centers = torch.rand(num_classes, dim, device=device)

    # collect stats to lists
    results = {
        'epochs': [],
        'L2_error': [],
    }

    suffix = get_suffix(args)

    if not args.verbose:
        kbar = pkbar.Kbar(target=args.num_epochs, width=50, always_stateful=True)

    for epoch in range(args.num_epochs):

        train_obj, centers = iteration(dataset_args, centers)

        results['epochs'].append(epoch)
        results['L2_error'].append(train_obj)

        if not args.verbose:
            kbar.update(epoch+1, values=[("L2 error", train_obj)])

        if str(train_obj) == 'nan' or train_obj > 1e16:
            break

    with open('./results/distributed_kmeans/' + 'results_' + suffix + '.pkl', 'wb') as filehandle:
        pickle.dump(results, filehandle)
