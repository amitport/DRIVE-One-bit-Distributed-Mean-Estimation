
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import pickle
import pkbar

import dataset_factories
import model_factories

from defs import get_device
from multiprocessing import freeze_support

import sys
sys.path.append('./compression')

from rotated_compression import hadamard_compress, hadamard_decompress
from rotated_compression import kashin_compress, kashin_decompress
from rotated_compression import drive_compress, drive_decompress
from rotated_compression import drive_plus_compress, drive_plus_decompress

from compression import fedavg_compress, fedavg_decompress
from compression import terngrad_compress, terngrad_decompress
from compression import terngrad_clipped_compress, terngrad_clipped_decompress
from compression import terngrad_layered_clipped_compress, terngrad_layered_clipped_decompress
from compression import count_sketch_compress, count_sketch_merge_and_decompress

##############################################################################
##############################################################################

def get_suffix(args):
    return '{}_{}_{}_{}'.format(args.dataset, args.compression_alg, args.clients, str(args.lr).replace('.', ''))

##########################################################################
####################### Training #########################################
##########################################################################

def train(epoch):
    
    net.train()

    train_loss = 0
    correct = 0
    total = 0
    
    if args.verbose:
        print("\nTraining:")
        kbar = pkbar.Kbar(target=len(trainloader)-1, epoch=epoch-1, num_epochs=args.num_epochs, width=50, always_stateful=True)
    
    compressed_client_grad_vecs = {}

    for batch_idx, (client_inputs, client_targets) in enumerate(trainloader):

        client_index = batch_idx % args.clients
        client_inputs, client_targets = client_inputs.to(device), client_targets.to(device)

        ### every client has the same batch size
        if len(client_inputs) < args.clientBatchSize:
            if args.verbose:
                kbar.update(batch_idx, values=[("loss", train_loss / (batch_idx + 1)), ("accuracy", 100. * correct / total)])
            continue

        optimizer.zero_grad()
        client_outputs = net(client_inputs)
        client_loss = criterion(client_outputs, client_targets)
        client_loss.backward()

        ##############################################################
        ################## extract client gradient ###################
        ##############################################################

        client_grad_vec = []
        for param in net.parameters():
            x = param.grad.view(-1)
            client_grad_vec.append(x)

        client_grad_vec = torch.cat(client_grad_vec)

        ##############################################################
        ############## update client stats ###########################
        ##############################################################

        train_loss += client_loss.item()
        _, predicted = client_outputs.max(1)
        total += client_targets.size(0)
        correct += predicted.eq(client_targets).sum().item()

        ##############################################################
        ############## compress client gradients #####################
        ##############################################################

        if args.compression_alg in rotated_algorithms:
            if need_padding:
                padded_client_grad_vec = torch.zeros(padded_dimension, device=device)
                padded_client_grad_vec[:params['dimension']] = client_grad_vec
                compressed_client_grad_vec = compress(padded_client_grad_vec, padded_dimension,
                                                      prng=snd_prngs[client_index])
            else:
                compressed_client_grad_vec = compress(client_grad_vec, padded_dimension,
                                                      prng=snd_prngs[client_index])
        elif args.compression_alg != 'count_sketch':
            compressed_client_grad_vec = compress(client_grad_vec, params=params, prng=snd_prngs[client_index])
        else:
            compressed_client_grad_vec = compress(client_grad_vec, params=params)

        ##############################################################
        ############## append to clients gradient list £##############
        ##############################################################

        compressed_client_grad_vecs[client_index] = compressed_client_grad_vec

        ##############################################################
        ##############  finished a pass over all clients #############
        ##############################################################

        grad_vec = torch.zeros(params['dimension'], device=device)

        ##############################################################
        ##############  progress bar #################################
        ##############################################################
        
        if args.verbose:
            kbar.update(batch_idx, values=[("loss", train_loss / (batch_idx + 1)), ("accuracy", 100. * correct / total)])
        
        ##############################################################
        ############## all clients finished a batch? #################
        ##############################################################
        
        if client_index == args.clients - 1:

            ##################################################################
            ####### compute the total gradient: merge and decompress #########
            ##################################################################

            if args.compression_alg != 'count_sketch':

                for compressed_client_index, compressed_client_grad_vec in compressed_client_grad_vecs.items():

                    ccgv, ccgv_metadata = compressed_client_grad_vec

                    if args.compression_alg in rotated_algorithms:
                        grad_vec += decompress(ccgv, padded_dimension, ccgv_metadata,
                                               prng=rcv_prngs[compressed_client_index])[:params['dimension']]
                    else:
                        grad_vec += decompress(ccgv, ccgv_metadata, params, prng=rcv_prngs[compressed_client_index])

                grad_vec /= args.clients

            else:
                grad_vec = decompress(compressed_client_grad_vecs, params)

            ##################################################################
            ########## make sure dict is clean for next round ################
            ##################################################################

            compressed_client_grad_vecs = {}

            ##################################################################
            ################## write modified gradient #######################
            ##################################################################

            offset = 0
            for param in net.parameters():
                slice_size = len(param.grad.view(-1))
                grad_slice = grad_vec[offset:offset + slice_size]
                offset += slice_size

                y = torch.Tensor(grad_slice.to('cpu')).resize_(param.grad.shape)
                param.grad = y.to(device).clone()

            ##################################################################
            ################## step ##########################################
            ##################################################################

            optimizer.step()

    ### return acc
    epoch_train_acc = 100. * correct / total

    return epoch_train_acc


##########################################################################
####################### Testing ##########################################
##########################################################################

def test(epoch):
    
    global best_acc

    net.eval()

    test_loss = 0
    correct = 0
    total = 0
    
    if args.verbose:
        print("Testing:")
        kbar = pkbar.Kbar(target=len(testloader)-1, epoch=epoch-1, num_epochs=args.num_epochs, width=50,
                          always_stateful=True)

    with torch.no_grad():
        
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if args.verbose:
                kbar.update(batch_idx, values=[("loss", test_loss / (batch_idx + 1)), ("accuracy", 100. * correct / total)])

    # save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc or epoch == start_epoch + args.num_epochs:
        if args.verbose:
            print('Saving checkpoint.')

        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }

        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt_{}.pth'.format(suffix))
        best_acc = acc

    ### return acc
    epoch_test_acc = 100. * correct / total

    return epoch_test_acc


if __name__ == '__main__':

    freeze_support()

    parser = argparse.ArgumentParser(description="""
        Simulation of distributed CNN with different compression algorithms.
        The results can be found in the results/distributed_cnn folder.
        The results filename is: results_<dataset>_<compression_alg>_<clients>_<lr (without ".")>.pkl""",
                                     formatter_class=argparse.RawTextHelpFormatter)

    ### learning rate
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate. Recommended values: 0.1 for CIFAR10, and 0.05 for CIFAR100')

    ### training epochs
    parser.add_argument('--num_epochs', default=250, type=int,
                        help='Number of epochs to run. Recommended: 150 for CIFAR10, and 250 for CIFAR100')

    ### resume training
    parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')

    ### clients and batch sizes
    parser.add_argument('--clients', default=10, type=int, help='Number of clients')
    parser.add_argument('--clientBatchSize', default=128, type=int, help='Batch size of a client')

    ### compression scheme
    parser.add_argument('--compression_alg', default='fedavg',
                        choices=['fedavg', 'terngrad_vanilla', 'count_sketch', 'hadamard', 'kashin', 'drive',
                                 'drive_plus'], help='compression/decompression scheme')

    ### dataset
    parser.add_argument('--dataset', default='CIFAR10', choices=['CIFAR10', 'CIFAR100'], help='')

    ### network
    parser.add_argument('--net', default='ResNet9', choices=['ResNet9', 'ResNet18'],
                        help='Use ResNet9 for CIFAR10, and ResNet18 for CIFAR100.')

    ### GPU index to work with, if more than one is available.
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU index to run the simulation (if more than one GPU is available)')

    ### seed
    parser.add_argument('--seed', default=42, type=int, help='Seed to use by all packages and PRNGs')

    ### print info to screen
    parser.add_argument('--verbose', default=False, action="store_true", help='Print more info')

    args = parser.parse_args()

    # print simulation info
    print ('*** Running distributed CNN with {} over {} clients with {} dataset and {} CNN model with learning rate of '
           '{} over {} epochs'.format(args.compression_alg, args.clients, args.dataset, args.net, args.lr, args.num_epochs))

    rotated_algorithms = ['hadamard', 'kashin', 'drive', 'drive_plus']

    ##########################################################################
    ##########################################################################

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    ##########################################################################
    ##########################################################################

    device, device_ids = get_device(args.gpu)

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    ##########################################################################
    ####################### Preparing data ###################################
    ##########################################################################

    print('==> Preparing data..')

    num_classes, trainset, trainloader, testset, testloader = getattr(dataset_factories, args.dataset)(
        args.clientBatchSize)

    ##########################################################################
    ########################## Net ###########################################
    ##########################################################################

    print('==> Building model..')
    net = getattr(model_factories, args.net)(num_classes)

    ##########################################################################
    ########################## Cuda ##########################################
    ##########################################################################

    net = net.to(device)
    if 'cuda' in device:
        net = torch.nn.DataParallel(net, device_ids=device_ids)
        cudnn.benchmark = True

    ##########################################################################
    ########################## Trainable parameters ##########################
    ##########################################################################

    pytorch_total_params = sum(p.numel() for p in net.parameters())
    pytorch_total_params_trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)

    if (pytorch_total_params != pytorch_total_params_trainable):
        raise Exception("pytorch_total_params != pytorch_total_params_trainable")

    ##########################################################################
    ########################## Parameters ####################################
    ##########################################################################

    params = {}

    params['dimension'] = pytorch_total_params_trainable
    params['gradLayerLengths'] = []
    params['gradLayerDims'] = []

    for param in net.parameters():
        params['gradLayerDims'].append(param.size())
        params['gradLayerLengths'].append(len(param.view(-1)))

    padded_dimension = params['dimension']
    if not params['dimension'] & (params['dimension'] - 1) == 0:
        padded_dimension = int(2 ** (np.ceil(np.log2(params['dimension']))))
    need_padding = padded_dimension != params['dimension']

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
    ############################### scheme ###################################
    ##########################################################################

    if args.compression_alg == 'fedavg':

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

    ##########################################################################

    elif args.compression_alg == 'terngrad_vanilla':

        compress = terngrad_compress
        decompress = terngrad_decompress

    ##########################################################################

    elif args.compression_alg == 'terngrad_clipped':

        compress = terngrad_clipped_compress
        decompress = terngrad_clipped_decompress

    ##########################################################################

    elif args.compression_alg == 'terngrad_layered_clipped':

        compress = terngrad_layered_clipped_compress
        decompress = terngrad_layered_clipped_decompress

    ##########################################################################

    elif args.compression_alg == 'count_sketch':

        compress = count_sketch_compress
        decompress = count_sketch_merge_and_decompress

    ##########################################################################

    else:

        raise Exception('compression algorithm undefined. Received {}'.format(args.compression_alg))

    ##########################################################################
    ########################## Params ########################################
    ##########################################################################

    if args.compression_alg == 'count_sketch':
        first_prime_after_dim_scaled = {
            'ResNet9': 153247,
            'ResNet18': 349187
        }

        params['first_prime_after_dim_scaled'] = first_prime_after_dim_scaled[args.net]
        params['rows'] = 1

    ##########################################################################
    ####################### Resume? ##########################################
    ##########################################################################

    if args.resume:
        
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'

        checkpoint = torch.load('./checkpoint/ckpt_{}.pth'.format(get_suffix(args)))
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    ##########################################################################
    ####################### Run ##############################################
    ########################################################################## 

    suffix = get_suffix(args)

    # collect stats to lists
    results = {
        'epochs': [],
        'trainACCs': [],
        'testACCs': [],
    }

    if not os.path.isdir('results'):
        os.mkdir('results')

    if not os.path.isdir('results/distributed_cnn'):
        os.mkdir('results/distributed_cnn')

    if not args.verbose:
        kbar = pkbar.Kbar(target=args.num_epochs, width=50, always_stateful=True)

    for epoch in range(start_epoch + 1, start_epoch + 1 + args.num_epochs):

        train_acc = train(epoch)
        test_acc = test(epoch)

        results['epochs'].append(epoch)
        results['trainACCs'].append(train_acc)
        results['testACCs'].append(test_acc)

        if not args.verbose:
            kbar.update(epoch, values=[("Train accuracy", train_acc), ("Test accuracy", test_acc)])

    with open('./results/distributed_cnn/' + 'results_' + suffix + '.pkl', 'wb') as filehandle:
        pickle.dump(results, filehandle)
