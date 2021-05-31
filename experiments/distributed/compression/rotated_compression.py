
import torch
import numpy as np
from geotorch import so

##############################################################################
##############################################################################

class RandomRotation:
    
    def __init__(self, dim, device='cpu'):  
        
        self.R = torch.zeros((dim,dim)).to(device)
        self.R = so.uniform_init_(self.R)
        
    def rotate(self, vec):
        return torch.matmul(self.R, vec)
    
    def inv_rotate(self, rvec):
        return torch.matmul(rvec, self.R) 

##############################################################################
##############################################################################

def one_dimentional_two_means(vec, dim, niters):
    
    vec_sum = vec.sum()
    
    old_assignments = torch.lt(vec, vec_sum / dim)
    old_size1 = -1
    
    size1 = old_assignments.sum()
    size2 = dim - size1
    
    sum1 =  vec[old_assignments].sum()
    sum2 =  vec_sum - sum1
    
    center1 = sum1 / size1
    center2 = sum2 / size2
    
    for i in range(niters):      
           
        old_size1 = size1.clone()  
        
        mid = (center1+center2)/2     
        assignments = torch.lt(vec, mid) 
        
        diff_1 = assignments.int() - old_assignments.int()        
        old_assignments = assignments
        
        size1 += diff_1.sum()
        size2 = dim - size1
        
        sum1 +=  (vec * diff_1).sum()
        sum2 =  vec_sum - sum1
        
        center1 = sum1 / size1
        center2 = sum2 / size2
        
        if old_size1 == size1:
            break
    
    return assignments, (center1, center2), (size1, size2)

##############################################################################
##############################################################################

##### in-place hadamard transform - 1D tensor

def hadamard_rotate(vec, dim):

    h = 2
    
    while h <= dim:
        
        hf = h // 2
        vec = vec.view(dim//h,h)
    
        vec[:,:hf]  = vec[:,:hf] + vec[:,hf:2*hf]
        vec[:,hf:2*hf] = vec[:,:hf] - 2*vec[:,hf:2*hf]
        h *= 2   
        
    vec = vec.view(-1)
    
##############################################################################
##############################################################################

def hadamard_compress(vec, dim, prng=None):

    ### in-place hadamard transform
    if prng is not None:
        vec = vec * (2 * torch.bernoulli(torch.ones(dim, device=vec.device) / 2, generator=prng) - 1) / np.sqrt(dim)
    hadamard_rotate(vec, dim)
        
    ### 1-bit stochastic quantization
    minimum = vec.min()
    maximum = vec.max()
    p = (vec-minimum)/(maximum-minimum)
    
    ### return the sign vector and the scaling parameters
    return torch.bernoulli(p), (minimum, maximum)

##############################################################################

def hadamard_decompress(vec, dim, metadata, prng=None):
    
    ### restore scale
    minimum, maximum = metadata
    vec = minimum + (maximum-minimum)*vec
        
    ### in-place hadamard transform (inverse)
    hadamard_rotate(vec, dim)
    if prng is not None:
        vec = vec * (2 * torch.bernoulli(torch.ones(dim, device=vec.device) / 2, generator=prng) - 1) / np.sqrt(dim)

    return vec    

##############################################################################
##############################################################################

def kashin_padded_dim(dim, pad_threshold):
    
    padded_dim = dim
    
    if not dim & (dim-1) == 0:
        padded_dim = int(2**(np.ceil(np.log2(dim))))
        if dim / padded_dim > pad_threshold:
            padded_dim = 2*padded_dim
    else:
        padded_dim = 2*dim
        
    return padded_dim

##############################################################################

def kashin_compress(vec, dim, metadata={'eta': 0.9, 'delta': 1.0, 'pad_threshold': 0.85, 'niters': 3}, prng=None):
    
    ### double dimension if required
    padded_dim = kashin_padded_dim(dim, metadata['pad_threshold'])
    
    ### Rademacher random variables
    if prng is not None:
        D = (2*torch.bernoulli(torch.ones(padded_dim, device=vec.device)/2, generator=prng)-1) / np.sqrt(padded_dim) 
    
    ### compute kashin's representation        
    kashin_coefficients = torch.zeros(padded_dim, device=vec.device)
    padded_x = torch.zeros(padded_dim, device=vec.device)
    
    M = torch.norm(vec) / np.sqrt(metadata['delta'] * padded_dim)
    
    for i in range(metadata['niters']):

        padded_x[:] = 0
        padded_x[:dim] = vec  
        
        ### in-place hadamard transform
        if prng is not None:
            padded_x = padded_x * D 
        hadamard_rotate(padded_x, padded_dim)
        
        b = padded_x   
        b_hat = torch.clamp(b, min=-M, max=M)
                
        kashin_coefficients = kashin_coefficients + b_hat
        
        ### in-place hadamard transform (inverse)
        hadamard_rotate(b_hat, padded_dim)
        if prng is not None:
            b_hat = b_hat * D
        
        vec = vec - b_hat[:dim]
                
        M = metadata['eta'] * M
    
    ### 1-bit stochastic quantization   
    minimum = kashin_coefficients.min()
    maximum = kashin_coefficients.max()  
    p = (kashin_coefficients-minimum)/(maximum-minimum)

    ### return the sign vector and the scaling parameters
    return torch.bernoulli(p), (minimum, maximum)
      
##############################################################################

def kashin_decompress(vec, dim, metadata, prng=None):
    
    ### padded dim
    padded_dim = vec.size()[0]
    
    ### restore scale
    minimum, maximum = metadata
    vec = minimum + (maximum-minimum)*vec
    
    ### Rademacher random variables
    if prng is not None:
        D = (2*torch.bernoulli(torch.ones(padded_dim, device=vec.device)/2, generator=prng)-1) / np.sqrt(padded_dim)  
    
    ### in-place hadamard transform
    hadamard_rotate(vec, padded_dim)
    if prng is not None:
        vec = vec * D
    
    return vec[:dim]   
       
##############################################################################
##############################################################################

def drive_compress(vec, dim, prng=None):

    ### in-place hadamard transform
    if prng is not None:
        vec = vec * (2 * torch.bernoulli(torch.ones(dim, device=vec.device) / 2, generator=prng) - 1) / np.sqrt(dim)
    hadamard_rotate(vec, dim)
            
    #### compute the scale (rotation preserves the L2 norm)
    scale = torch.norm(vec,2)**2 / torch.norm(vec,1)

    ##### take the sign
    vec = 1.0 - 2 * ( vec < 0 ) 
    
    #### send
    return vec, scale

##############################################################################

def drive_decompress(vec, dim, scale, prng=None):
        
    ### in-place hadamard transform (inverse)
    hadamard_rotate(vec, dim)
    if prng is not None:
        vec = vec * (2 * torch.bernoulli(torch.ones(dim, device=vec.device) / 2, generator=prng) - 1) / np.sqrt(dim)

    ##### scale and return   
    return scale * vec 

##############################################################################
##############################################################################

def drive_plus_compress(vec, dim, metadata={'niters': 3}, prng=None):
    
    ### in-place hadamard transform  
    if prng is not None:
        vec = vec * (2 * torch.bernoulli(torch.ones(dim, device=vec.device) / 2, generator=prng) - 1) / np.sqrt(dim)
    hadamard_rotate(vec, dim)
                
    ##### finding the centroids
    assignments, centers, sizes = one_dimentional_two_means(vec, dim, metadata['niters'])
            
    ### compute the scale (rotation preserves the L2 norm)    
    scale = torch.norm(vec,2)**2 / (sizes[0]*centers[0]**2 + sizes[1]*centers[1]**2)
        
    #### scale and send        
    return assignments, (scale*centers[0], scale*centers[1])

##############################################################################

def drive_plus_decompress(assignments, dim, centers, prng=None):
    
    vec = torch.zeros(dim, device=assignments.device)

    vec[assignments] = centers[0]
    vec[~assignments] = centers[1]
            
    ### in-place hadamard transform (inverse)
    hadamard_rotate(vec, dim)
    if prng is not None:
        vec = vec * (2 * torch.bernoulli(torch.ones(dim, device=vec.device) / 2, generator=prng) - 1) / np.sqrt(dim)

    return vec 

##############################################################################
##############################################################################

def drive_urr_compress(vec, dim, urr):
    
    ### rotate
    vec = urr.rotate(vec)
        
    #### compute the scale (rotation preserves the L2 norm)
    scale = torch.norm(vec,2)**2 / torch.norm(vec,1)

    ##### take the sign
    vec = 1.0 - 2 * ( vec < 0 ) 
    
    #### send
    return vec, scale

##############################################################################

def drive_urr_decompress(vec, dim, scale, urr):
    
    ### inverse rotate    
    vec = urr.inv_rotate(vec)
        
    ##### scale and return   
    return scale * vec 

##############################################################################
##############################################################################

def drive_urr_plus_compress(vec, dim, urr, niters=3, prng=None):
    
    ### rotate
    vec = urr.rotate(vec)
                
    ##### finding the centroids
    assignments, centers, sizes = one_dimentional_two_means(vec, dim, niters)
            
    ### compute the scale (rotation preserves the L2 norm)    
    scale = torch.norm(vec,2)**2 / (sizes[0]*centers[0]**2 + sizes[1]*centers[1]**2)
        
    #### scale and send        
    return assignments, (scale*centers[0], scale*centers[1])

##############################################################################

def drive_urr_plus_decompress(assignments, dim, centers, urr, prng=None):
    
    vec = torch.zeros(dim, device=assignments.device)

    vec[assignments] = centers[0]
    vec[~assignments] = centers[1]
            
    ### inverse rotate    
    vec = urr.inv_rotate(vec)
                    
    return vec 

##############################################################################
##############################################################################
