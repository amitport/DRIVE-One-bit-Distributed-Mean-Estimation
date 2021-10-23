#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pickle
import traceback
import argparse
import pandas as pd

import matplotlib_inline.backend_inline

matplotlib_inline.backend_inline.set_matplotlib_formats('pdf')

plt.rcParams["font.family"] = "Verdana"
plt.rcParams["font.sans-serif"] = "Verdana"

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rc('text', usetex=True)

##############################################################################
##############################################################################

df = pd.read_pickle("./entropy/log_normal.pkl")

alg_minmax = ['sq_{}b_minmax'.format(i) for i in range(1, 13)]
alg_sqrt2norm = ['sq_{}b_sqrt2norm'.format(i) for i in range(1, 13)]

df_minmax = df[alg_minmax]
df_sqrt2norm = df[alg_sqrt2norm]

d = 2 ** 19

ddf_minmax = df_minmax.loc[d]
ddf_sqrt2norm = df_sqrt2norm.loc[d]

l_bits_minmax = []
l_nmse_minmax = []

l_bits_sqrt2norm = []
l_nmse_sqrt2norm = []

for row_minmax, row_sqrt2norm in zip(ddf_minmax, ddf_sqrt2norm):
  bits_minmax, nmse_minmax = tuple(map(float, row_minmax.split(', ')))
  bits_sqrt2norm, nmse_sqrt2norm = tuple(map(float, row_sqrt2norm.split(', ')))

  l_bits_minmax.append(bits_minmax)
  l_nmse_minmax.append(nmse_minmax)

  l_bits_sqrt2norm.append(bits_sqrt2norm)
  l_nmse_sqrt2norm.append(nmse_sqrt2norm)

##############################################################################
##############################################################################

plt.figure(figsize=(12, 6))

n = 1
for x, y in zip(l_bits_minmax, l_nmse_minmax):
  plt.text(x + 0.02, y, str(n), color="darkgreen", fontsize=18)
  n += 1

n = 1
for x, y in zip(l_bits_sqrt2norm, l_nmse_sqrt2norm):
  plt.text(x - 0.04, y, str(n), color="blue", fontsize=18)
  n += 1

plt.xlim([0.9, 2.1])
plt.ylim([3 * 10 ** -3, 100])

fs = 16

plt.semilogy(l_bits_minmax, l_nmse_minmax, marker='P', linestyle='--', markersize=12, color='darkgreen',
             label='SQ + Huffman')
plt.semilogy(l_bits_sqrt2norm, l_nmse_sqrt2norm, marker='D', linestyle='--', markersize=12, color='blue',
             label='Enhanced SQ + Huffman')
plt.semilogy([0] + l_bits_minmax, [0.0571 for i in range(len([0] + l_bits_minmax))], linestyle='-', color='black',
             label='DRIVE (1b per-coordiante)')

plt.ylabel('NMSE', fontsize=fs)
plt.xlabel('bits/coordinate', fontsize=fs)

plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)

plt.legend(fontsize=fs)
plt.grid()
plt.show()
