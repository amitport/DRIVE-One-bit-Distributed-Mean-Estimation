
import torch

def get_device(gpu):
    if gpu is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device_ids = [x for x in range(torch.cuda.device_count())]
    else:
        if gpu > torch.cuda.device_count()-1:
            raise Exception('got gpu index={}, but there are only {} GPUs'.format(gpu, torch.cuda.device_count()))
        if torch.cuda.is_available():
            device = 'cuda:{}'.format(gpu)
            device_ids = [gpu]
        else:
            device = 'cpu'

    if device == 'cpu':
        print('*** Warning: No GPU was found, running over CPU.')

    print('*** Set device to {}'.format(device))
    if device == 'cuda' and torch.cuda.device_count() > 1:
        print('*** Running on multiple GPUs ({})'.format(torch.cuda.device_count()))

    return device, device_ids
