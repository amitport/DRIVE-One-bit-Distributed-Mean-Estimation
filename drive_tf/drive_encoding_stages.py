import tensorflow as tf
from tensorflow_model_optimization.python.core.internal import tensor_encoding as te


def _calculate_scale(x, x_estimation):
  # projected scale is ||x||^2 / <x, x_estimation>
  return tf.norm(x) ** 2 / tf.reduce_sum(x * x_estimation)


def _randomize_zero_sign(x):
  zero_indices = tf.where(tf.equal(x, 0.))
  onebit_negative_as_zero = tf.cast(tf.where(tf.less_equal(x, 0.), 0., 1.), x.dtype)

  random_signs = tf.cast(
    tf.random.uniform(
      shape=tf.shape(zero_indices)[:1], minval=0, maxval=2, dtype=tf.int32),
    dtype=x.dtype
  )
  # onebit_signs is in {0, 1}
  onebit_signs = tf.tensor_scatter_nd_add(onebit_negative_as_zero, zero_indices, random_signs)
  # signs is in {-1, 1}
  signs = onebit_signs * 2. - 1.

  return onebit_signs, signs


@te.core.tf_style_encoding_stage
class DriveSignStage(te.core.EncodingStageInterface):
  ENCODED_VALUES_KEY = 'encoded'
  SCALE_KEY = 'scale'

  @property
  def name(self):
    """See base class."""
    return 'drive_sign'

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

    # We take the sign of the vector with randomized zeros in order to be unbiased.
    # We note that this probably has little effect in practice and we could have just
    # map all zeros to either 0 or 1. This is due to the fact that zeros are unlikely
    # after running the high-dimensional Hadamard rotation that precedes this stage
    onebit_signs, signs = _randomize_zero_sign(x)

    scale = _calculate_scale(x, signs)

    return {
      self.ENCODED_VALUES_KEY: onebit_signs,
      self.SCALE_KEY: scale,
    }

  def decode(self,
             encoded_tensors,
             decode_params,
             num_summands=None,
             shape=None):
    """See base class."""
    del decode_params, num_summands, shape  # Unused.

    onebit_signs = encoded_tensors[self.ENCODED_VALUES_KEY]
    scale = encoded_tensors[self.SCALE_KEY]

    signs = onebit_signs * 2. - 1.

    return scale * signs


@tf.function
def _two_means_1d_floyd(points):
  # simple floyd algorithm for 2-mean in 1d

  assignments = tf.random.uniform(shape=points.shape, minval=0, maxval=2, dtype=tf.int32)
  epsilon = tf.constant(1e-6, dtype=points.dtype)

  def assignments_to_centers(assignments):
    cluster_sums = tf.math.unsorted_segment_sum(points, assignments, num_segments=2)

    cluster_1_size = tf.reduce_sum(assignments)
    cluster_sizes = tf.cast(tf.stack([tf.size(points) - cluster_1_size, cluster_1_size]), dtype=points.dtype)

    return cluster_sums / (cluster_sizes + epsilon)

  def body(old_assignments, clusters_centers, stop_condition):
    new_assignments = tf.cast(tf.greater(points, (clusters_centers[0] + clusters_centers[1]) / 2), tf.int32)

    return [
      new_assignments,
      assignments_to_centers(new_assignments),
      # stop_condition -- continue while there is diff in assignment:
      tf.reduce_any(tf.cast(old_assignments - new_assignments, tf.bool)),
    ]

  def condition(assignments, clusters_centers, stop_condition):
    return stop_condition

  assignments, clusters_centers, stop_condition = tf.nest.map_structure(tf.stop_gradient, tf.while_loop(
    condition, body, loop_vars=[
      assignments,
      assignments_to_centers(assignments),
      True,
    ],
    parallel_iterations=1,
  ))

  return assignments, clusters_centers


@tf.function
def _split_at_mean_clustering(x):
  th = tf.reduce_mean(x)

  neg_mask = tf.less_equal(x, th)
  neg_com = tf.reduce_mean(tf.boolean_mask(x, neg_mask))

  pos_mask = tf.logical_not(neg_mask)  # same as tf.greater(x, th)
  pos_com = tf.reduce_mean(tf.boolean_mask(x, pos_mask))

  return pos_mask, [neg_com, pos_com]


@te.core.tf_style_encoding_stage
class DrivePlusClusteringStage(te.core.EncodingStageInterface):
  ENCODED_VALUES_KEY = 'encoded'
  CENTER_0_KEY = 'c0'
  CENTER_1_KEY = 'c1'

  def __init__(self, clustering: str = 'floyd'):
    """
    :param clustering: one of ['floyd', 'split_at_mean']
    """
    if clustering == 'floyd':
      self.clustering = _two_means_1d_floyd
    elif clustering == 'split_at_mean':
      self.clustering = _split_at_mean_clustering
    else:
      raise TypeError("'clustering' param must be one of ['floyd', 'split_at_mean']")

  @staticmethod
  def _inner_decode(assignments, center_0, center_1):
    return tf.where(tf.cast(assignments, tf.bool), center_1, center_0)

  @property
  def name(self):
    """See base class."""
    return 'drive_plus_clustering'

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

    assignments, centers = self.clustering(x)

    x_estimation = DrivePlusClusteringStage._inner_decode(assignments, centers[0], centers[1])
    scale = _calculate_scale(x, x_estimation)

    # scale centers
    centers *= scale

    return {
      self.ENCODED_VALUES_KEY: tf.cast(assignments, tf.float32),  # float is needed for bit-packing
      self.CENTER_0_KEY: centers[0],
      self.CENTER_1_KEY: centers[1],
    }

  def decode(self,
             encoded_tensors,
             decode_params,
             num_summands=None,
             shape=None):
    """See base class."""
    del decode_params, num_summands, shape  # Unused.

    return DrivePlusClusteringStage._inner_decode(
      encoded_tensors[self.ENCODED_VALUES_KEY],
      encoded_tensors[self.CENTER_0_KEY],
      encoded_tensors[self.CENTER_1_KEY])
