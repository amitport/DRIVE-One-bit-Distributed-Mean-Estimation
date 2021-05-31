
import torch
import sys
sys.path.append('../.')

##############################################################################
##############################################################################
##############################################################################

def fedavg_compress(vec, params=None, prng=None):      
    return vec, None

def fedavg_decompress(vec, metadata=None, params=None, prng=None):
    return vec

##############################################################################
##############################################################################
##############################################################################

def terngrad_compress(grad, metadata=None, params=None, prng=None): 
    
    ### extract max and abs
    grad_abs = grad.abs()
    grad_max = grad_abs.max()
    
    ### randomized rounding
    p = torch.clamp(grad_abs/grad_max, 0, 1)
    b = torch.bernoulli(p)
    
    ### ternarize (grad_max is "sent" separately)
    return grad.sign()*b, grad_max

def terngrad_decompress(grad, scale=None, params=None, prng=None):  
    return scale * grad

##############################################################################
##############################################################################
##############################################################################

def terngrad_clipped_compress(grad, metadata=None, params=None, prng=None): 
    
    tern_grad = torch.zeros(grad.size(), device=grad.device)
 
    ### clipping
    th = 2.5 * torch.std(grad)
    tern_grad = torch.clamp(grad, -th, th)
            
    ### ternarize
    compressed_grad = tern_grad.sign() * torch.bernoulli(tern_grad.abs() / tern_grad.abs().max())
            
    return compressed_grad, th

def terngrad_clipped_decompress(grad, scale=None, params=None, prng=None):  
    return scale * grad 

##############################################################################
##############################################################################
##############################################################################

def terngrad_layered_clipped_compress(grad, metadata=None, params=None, prng=None):  
    
    compressed_grad = torch.zeros(grad.size(), device=grad.device)
    compressed_grad_scaling = []

    offset = 0
    
    for l in params['gradLayerLengths']:
        
        ### taking a layer        
        grad_slice =  grad[offset:offset+l]
            
        ### clipping
        th = 2.5 * torch.std(grad_slice)
        clipped_grad_slice = torch.clamp(grad_slice, -th, th)
        th_scale = clipped_grad_slice.abs().max()
            
        ### ternarize
        compressed_grad_scaling.append(th_scale)
        compressed_grad[offset:offset+l] = clipped_grad_slice.sign() * torch.bernoulli(clipped_grad_slice.abs() / th_scale)
        
        ### move to next layer
        offset += l
            
    return compressed_grad, compressed_grad_scaling

def terngrad_layered_clipped_decompress(grad, metadata=None, params=None, prng=None):  
    
    offset = 0
    
    for scale, l in zip(metadata, params['gradLayerLengths']):   
            
        ### taking a layer        
        grad[offset:offset+l] =  scale * grad[offset:offset+l]
            
        ### move to next layer
        offset += l    
        
    return grad

##############################################################################
##############################################################################
##############################################################################


##############################################################################
    
def count_sketch_compress(grad, params=None):

    from csvec_extended import CSVec_Extended

    ### extract params
    d = int(params['dimension'])
    c = int(params['first_prime_after_dim_scaled']) 
    r = int(params['rows']) 

    sketch = CSVec_Extended(d=d, c=c, r=r, device=grad.device)
    sketch.accumulateVec(grad)
    return sketch.to_1d_tensor()

def count_sketch_merge_and_decompress(compressed_client_vecs, params):  

    from csvec_extended import CSVec_Extended

    result_sketch_1d_tensor = sum(compressed_client_vecs.values())
    
    ### extract params
    d = int(params['dimension'])
    c = int(params['first_prime_after_dim_scaled'])
    r = int(params['rows']) 
    
    result_sketch = CSVec_Extended(d=d, c=c, r=r, device=result_sketch_1d_tensor.device)
    result_sketch.from_1d_tensor(result_sketch_1d_tensor)
    vec = torch.div(result_sketch.unSketch(k=d), len(compressed_client_vecs))  
    return vec
         
##############################################################################
##############################################################################
##############################################################################
