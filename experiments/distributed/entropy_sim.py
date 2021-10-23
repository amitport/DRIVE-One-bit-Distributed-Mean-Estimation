import argparse
import math
from functools import partial
from itertools import product
from pathlib import Path
import pandas as pd
import huffman
import torch
import pkbar

from experiments.distributed.run_distributed_speed_error import compute_error


def sq_compress(vec, bits, normalization):
  minimum = vec.min()
  maximum = vec.max()

  k = 2 ** bits
  if normalization == 'minmax':
    s = maximum - minimum
  elif normalization == 'sqrt2norm':
    l2_norm = torch.linalg.norm(vec, ord=2)
    s = math.sqrt(2) * l2_norm
  else:
    raise RuntimeError(f'Unexpected \'normalization\' value: {normalization}')

  r_float = ((vec - minimum) / s) * (k - 1)

  r = torch.floor(r_float)
  p = r_float - r
  r = r + torch.bernoulli(p)
  return r, s, minimum


def sq_decompress(vec, s, minimum, bits):
  k = 2 ** bits
  return (vec * s) / (k - 1) + minimum


def huffman_bit_count(vec):
  freq = [(str(i), int(f)) for i, f in enumerate(torch.bincount(vec.int()))]
  codelenghts = {k: len(code) for k, code in huffman.codebook(freq).items()}

  return sum([codelenghts[k] * f for k, f in freq])


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="""
        Generates the data for entropy table in the paper.
        The results are saved as a pandas dataframe (rows are dimensions / columns are algorithm names).
        The results folder is specified by the <path> parameter""",
                                   formatter_class=argparse.RawTextHelpFormatter)

  ### verbosity
  parser.add_argument('--verbose', default=True, action='store_true', help='detailed progress')

  ### GPU index to work with, if more than one is available.
  parser.add_argument('--gpu', default='0', type=str, help='gpu index to run the simulation')

  ### seed
  parser.add_argument('--seed', default=42, type=int, help='random seed')

  ### path to result folder
  parser.add_argument('--path', default='./results/entropy', type=str, help='random seed')

  args = parser.parse_args()

  ##########################################################################
  ##########################################################################

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  if device == 'cuda' and torch.cuda.device_count() > 1:
    device = f'device:{args.gpu}'
  print(f'running on device: {device}')

  torch.manual_seed(args.seed)

  vec_distributions = [torch.distributions.LogNormal(0, 1)]#,
                       # torch.distributions.Normal(0, 1),
                       # torch.distributions.exponential.Exponential(torch.Tensor([1.0]))]

  path = args.path
  verbose = args.verbose

  ### create directory if needed
  Path(path).mkdir(parents=True, exist_ok=True)

  algorithms = [(f'sq_{bits}b_{normalization}',
                  partial(sq_compress, bits=bits, normalization=normalization),
                  partial(sq_decompress, bits=bits)
                 ) for
                bits, normalization
                in product(range(1, 13), ['minmax', 'sqrt2norm'])]

  dimensions = [128, 8192, 524288, 33554432]

  trials_per_algdim = {
    128 : 100,
    8_192 : 100,
    524_288 : 10,
    33_554_432 : 10,
  }
  n_clients = 10

  for vec_dist in vec_distributions:
    vec_dist_name = str(type(vec_dist)).split(".")[-2]

    df = pd.DataFrame(columns=[algname for algname, alg_compress, alg_decompress in algorithms])
    for dim in dimensions:
      df = df.append(pd.Series(name=dim))

    for dim in dimensions:
      for algname, alg_compress, alg_decompress in algorithms:

        print(f"\n*** Running {vec_dist_name=} {algname=} {dim=}")

        trial_num = trials_per_algdim[dim]
        if verbose:
          print('\n')
          kbar = pkbar.Kbar(target=trial_num, width=50, always_stateful=True)

        total_bit_widths = 0.
        total_error = 0.
        for trial in range(trial_num):
          if verbose:
            kbar.update(trial)

          original_vec = vec_dist.sample([dim]).to(device).view(-1)
          reconstructed_vec = torch.zeros(dim, device=device)
          for _ in range(n_clients):
            compressed_vec, s, minimum = alg_compress(original_vec)
            entropy_bits = huffman_bit_count(compressed_vec)
            total_bit_widths += entropy_bits / dim

            reconstructed_vec += alg_decompress(compressed_vec, s, minimum)

          total_error += compute_error(original_vec, reconstructed_vec / n_clients)

        error = total_error / trial_num
        bit_width = total_bit_widths / (trial_num * n_clients)

        df.loc[dim].at[algname] = f"{bit_width:.8f}, {error:.8f}"

        if verbose:
          print(f"\n\n==> {bit_width=:.8f}, {error=:.8f}")

    df.to_pickle(f"{path}/{vec_dist_name}.pkl")
