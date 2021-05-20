from tensorflow_model_optimization.python.core.internal import tensor_encoding as te

from drive_tf import drive_encoder, drive_plus_encoder
from experiments.federated.kashin_onebit_encoder import kashin_onebit_encoder
from experiments.federated.terngrad_encoder import terngrad_encoder

ENCODERS = {
    'hadamard_quantization': te.encoders.hadamard_quantization,
    'kashin_quantization': kashin_onebit_encoder,
    'terngrad': terngrad_encoder,
    'drive': drive_encoder,
    'drive_plus': drive_plus_encoder,
}