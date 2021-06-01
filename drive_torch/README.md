# DRIVE: PyTorch implementation

### install requirements:

```setup
pip install -r requirements.txt
```

### usage 

```python
from drive_torce import drive_compress, drive_decompress, drive_plus_compress, drive_plus_decompress
import torch

# sender and receiver should use the same seed:
seed = 42

sgen = torch.Generator()
sgen.manual_seed(seed)

rgen = torch.Generator()
rgen.manual_seed(seed)

# a pytorch vector padded with zeros to have power of 2 elements
vec = ...

s_vec, scale = drive_compress(vec, prng=sgen)
decompressed_vec = drive_decompress(s_vec, scale, prng=rgen)

# OR if using drive_plus:
# s_vec, (scale0, scale1) = drive_plus_compress(vec, prng=sgen)
# decompressed_vec = drive_plus_decompress(s_vec, (scale0, scale1), prng=rgen)
```

See `experiments/distributed` for additional usage examples.