import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as tdt

import numpy as np

# MNIST dataset for distributed algorithms (e.g., K-means and power iteration)
def MNIST_DA(num_clients):
    transform_train = transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    transform_test = transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)

    dataset = tdt.ConcatDataset([trainset, testset])

    dataset_len = len(dataset)
    batch_size = int(dataset_len/num_clients)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False)

    dim = np.prod(dataset[0][0].shape)

    dataset_labels = [dataset[i][1] for i in range(dataset_len)]
    dataset_features = [dataset[i][0] for i in range(dataset_len)]

    num_classes = len(np.unique(dataset_labels))

    return dataloader, dataset_features, dataset_labels, dataset_len, num_classes, batch_size, dim

# CIFAR10 dataset for distributed algorithms (e.g., K-means and power iteration)
def CIFAR10_DA(num_clients):
        
    transform_data = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_data)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_data)
    
    dataset = tdt.ConcatDataset([trainset, testset])
    
    dataset_len = len(dataset)
    batch_size = int(dataset_len/num_clients)
        
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False)
    
    dim = np.prod(dataset[0][0].shape)
    
    dataset_labels = [dataset[i][1] for i in range(dataset_len)]
    dataset_features = [dataset[i][0] for i in range(dataset_len)]
    
    num_classes = len(np.unique(dataset_labels))

    return dataloader, dataset_features, dataset_labels, dataset_len, num_classes, batch_size, dim  


def CIFAR10(batch_size):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)
    
    num_classes = 10

    return num_classes, trainset, trainloader, testset, testloader


def CIFAR100(batch_size):
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)
    
    num_classes = 100

    return num_classes, trainset, trainloader, testset, testloader

def MNIST(all_clients_batch_size):
    
    transform_train = transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])

    transform_test = transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    trainset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform_train)   
    trainloader = torch.utils.data.DataLoader(
      trainset, batch_size=all_clients_batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)
    
    num_classes = 10

    return num_classes, trainset, trainloader, testset, testloader

