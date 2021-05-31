
import os
import sys
sys.path.append('./results')
from plot import label

##############################################################################
##############################################################################

def get_cuda(cuda):
    if cuda is not None:
        return '--gpu {}'.format(cuda)
    else:
        return ''

##############################################################################
##############################################################################

cuda = None #if multiple GPU are available, one can set the GPU index in order to run over a specific GPU device 
epochs = 200
clients = [10, 100, 1000]
datasets = ['MNIST_DA', 'CIFAR10_DA']

##############################################################################
##############################################################################

compression_algs = ['baseline', 'hadamard', 'kashin', 'drive', 'drive_plus']

##############################################################################
##############################################################################
for dataset in datasets:
    for compression_scheme in compression_algs:
        for clients_num in clients:
            command = 'python3 distributed_power_iteration.py --dataset {} --clients {} --compression_alg {} ' \
                      '--num_epochs {} {}'.format(dataset, clients_num, compression_scheme, epochs, get_cuda(cuda))

            print('\n\nRunning {} by: $ {}'.format(label[compression_scheme], command))
            os.system(command)

