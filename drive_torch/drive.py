import torch
import numpy as np

##############################################################################
##############################################################################

def hadamard_rotate(vec):
  '''
  In-place 1D hadamard transform 
  '''
    
  numel = vec.numel()
  if (numel & (numel-1) == 0) and numel != 0:
      raise Exception("vec numel must be a power of 2")
      
  h = 2

  while h <= numel:
      
    hf = h // 2
    vec = vec.view(numel // h, h)

    vec[:, :hf] = vec[:, :hf] + vec[:, hf:2 * hf]
    vec[:, hf:2 * hf] = vec[:, :hf] - 2 * vec[:, hf:2 * hf]
    h *= 2

##############################################################################
##############################################################################

def one_dimentional_two_means(vec, niters):
  '''
  Simplified Lloyd's algorithm for 2-means and 1D data
  '''
    
  numel = vec.numel()

  vec_sum = vec.sum()

  old_assignments = torch.lt(vec, vec_sum / numel)

  size1 = old_assignments.sum()
  size2 = numel - size1

  sum1 = vec[old_assignments].sum()
  sum2 = vec_sum - sum1

  center1 = sum1 / size1
  center2 = sum2 / size2

  for i in range(niters):
      
    old_size1 = size1.clone()

    mid = (center1 + center2) / 2
    assignments = torch.lt(vec, mid)

    diff_1 = assignments.int() - old_assignments.int()
    old_assignments = assignments

    size1 += diff_1.sum()
    size2 = numel - size1

    sum1 += (vec * diff_1).sum()
    sum2 = vec_sum - sum1

    center1 = sum1 / size1
    center2 = sum2 / size2

    if old_size1 == size1:
      break

  return assignments, (center1, center2), (size1, size2)

##############################################################################
##############################################################################

def drive_compress(vec, prng=None):
  '''
  :param vec: the vector to compress (currently we require vec numel to be a power of two)
  :param prng: a generator that determines the specific (random) Hadamard rotation
  :return: compressed vector
  '''
  
  ### dimension
  numel = vec.numel()
  
  ### in-place hadamard transform
  if prng is not None:
    vec = vec * (2 * torch.bernoulli(torch.ones(numel, device=vec.device) / 2, generator=prng) - 1) / np.sqrt(numel)
  hadamard_rotate(vec, numel)

  #### compute the scale (rotation preserves the L2 norm)
  scale = torch.norm(vec, 2) ** 2 / torch.norm(vec, 1)

  ##### take the sign
  vec = 1.0 - 2 * (vec < 0)

  #### send
  return vec, scale

##############################################################################

def drive_decompress(vec, scale, prng=None):
  '''
    :param assignments: sign(Rx) from the paper
    :param scale: S from the paper
    :param prng: random generator for Hadamard rotation, should have the same state used for compression
    :return: decompressed vector
    '''

  ### dimension
  numel = vec.numel()

  ### in-place hadamard transform (inverse)
  hadamard_rotate(vec, numel)
  if prng is not None:
    vec = vec * (2 * torch.bernoulli(torch.ones(numel, device=vec.device) / 2, generator=prng) - 1) / np.sqrt(numel)

  ##### scale and return
  return scale * vec

##############################################################################
##############################################################################

def drive_plus_compress(vec, kmeans_niters=3, prng=None):
  '''
  :param vec: the vector to compress (currently we require vec numel to be a power of two)
  :param kmeans_niters: the number of Lloyd's K-means iterations
  :param prng: a generator that determines the specific (random) Hadamard rotation
  :return: compressed vector
  '''

  ### dimension
  numel = vec.numel()
  
  ### in-place hadamard transform
  if prng is not None:
    vec = vec * (2 * torch.bernoulli(torch.ones(numel, device=vec.device) / 2, generator=prng) - 1) / np.sqrt(numel)
  hadamard_rotate(vec, numel)

  ##### finding the centroids
  assignments, centers, sizes = one_dimentional_two_means(vec, numel, kmeans_niters)

  ### compute the scale (rotation preserves the L2 norm)
  scale = torch.norm(vec, 2) ** 2 / (sizes[0] * centers[0] ** 2 + sizes[1] * centers[1] ** 2)

  #### scale and send
  return assignments, (scale * centers[0], scale * centers[1])

##############################################################################

def drive_plus_decompress(assignments, centers, prng=None):
  '''
  :param assignments: c from the paper
  :param centers: c0, c1 from the paper
  :param prng: random generator for Hadamard rotation, should have the same state used for compression
  :return: decompressed vector
  '''
  
  ### dimension
  numel = assignments.numel()
  
  vec = torch.zeros(numel, device=assignments.device)
  vec[assignments] = centers[0]
  vec[~assignments] = centers[1]

  ### in-place hadamard transform (inverse)
  hadamard_rotate(vec, numel)
  if prng is not None:
    vec = vec * (2 * torch.bernoulli(torch.ones(numel, device=vec.device) / 2, generator=prng) - 1) / np.sqrt(numel)

  return vec

##############################################################################
##############################################################################

