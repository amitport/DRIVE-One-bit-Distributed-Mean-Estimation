
import os
import sys

sys.path.append('./results')
from plot import label


##############################################################################
##############################################################################

def resume_arg(resume):
    if resume:
        return '--resume'
    else:
        return ''


def get_cuda(cuda):
    if cuda is not None:
        return '--gpu {}'.format(cuda)
    else:
        return ''


##############################################################################
##############################################################################

resume = False
cuda = None #if multiple GPU are available, one can set the GPU index in order to run over a specific GPU device

epochs = {
    'CIFAR10': 150,
    'CIFAR100': 250,
}

clients = [10]

model = {
    'CIFAR10': 'ResNet9',
    'CIFAR100': 'ResNet18',
}

datasets = ['CIFAR10', 'CIFAR100']

##############################################################################
##############################################################################

compression_algs = ['fedavg', 'terngrad_vanilla', 'count_sketch', 'hadamard', 'kashin', 'drive', 'drive_plus']

lr = {
    'CIFAR10': 0.1,
    'CIFAR100': 0.05,
}

##############################################################################
##############################################################################

for dataset in datasets:
    for compression_scheme in compression_algs:
        for clients_num in clients:
            command = 'python3 distributed_cnn.py {} --dataset {} --clients {} --compression_alg {} --num_epochs {} ' \
                      '--lr {} --net {} {}'.format(resume_arg(resume), dataset, clients_num, compression_scheme,
                                                   epochs[dataset], lr[dataset], model[dataset], get_cuda(cuda))

            print('\n\nRunning {} by: $ {}'.format(label[compression_scheme], command))
            os.system(command)
