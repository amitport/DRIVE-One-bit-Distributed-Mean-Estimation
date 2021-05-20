import tensorflow as tf
from tensorflow_model_optimization.python.core.internal import tensor_encoding as te


@te.core.tf_style_encoding_stage
class TernGradStage(te.core.EncodingStageInterface):
  C = 2.5

  ENCODED_VALUES_KEY = 'encoded'
  ENCODED_SIGNS_KEY = 'signs'
  SCALE_KEY = 'scale'

  @property
  def name(self):
    """See base class."""
    return 'terngrad'

  @property
  def compressible_tensors_keys(self):
    """See base class."""
    return [self.ENCODED_VALUES_KEY]

  @property
  def commutes_with_sum(self):
    """See base class."""
    return False

  @property
  def decode_needs_input_shape(self):
    """See base class."""
    return False

  def get_params(self):
    """See base class."""
    return {}, {}

  def encode(self, x, encode_params):
    """See base class."""
    del encode_params  # Unused.

    # clipping
    std = tf.math.reduce_std(x)
    x = tf.clip_by_value(x, -self.C * std, self.C * std)

    # consider only absolute value
    abs_x = tf.math.abs(x)
    scale = tf.math.reduce_max(abs_x)

    h = tf.random.uniform(tf.shape(x), 0, scale)
    abs_quantization = tf.cast(tf.less(h, abs_x), x.dtype)

    return {
      self.ENCODED_VALUES_KEY: abs_quantization,
      self.ENCODED_SIGNS_KEY: tf.sign(x),
      self.SCALE_KEY: scale,
    }

  def decode(self,
             encoded_tensors,
             decode_params,
             num_summands=None,
             shape=None):
    """See base class."""
    del decode_params, num_summands, shape  # Unused.

    abs_quantization = encoded_tensors[self.ENCODED_VALUES_KEY]
    signs = encoded_tensors[self.ENCODED_SIGNS_KEY]
    scale = encoded_tensors[self.SCALE_KEY]

    return abs_quantization * signs * scale


def terngrad_encoder():
  return te.core.EncoderComposer(
    te.stages.BitpackingEncodingStage(1)
  ).add_parent(
    TernGradStage(),
    TernGradStage.ENCODED_VALUES_KEY
  ).add_parent(
    te.stages.FlattenEncodingStage(),
    te.stages.FlattenEncodingStage.ENCODED_VALUES_KEY
  ).make()
