import tensorflow as tf
import tensorflow_probability as tfp

LARGE_PRIME = 2 ** 61 - 1


# On a previous machine I had an issue with l2 cache size
# and had to use this function (left for posterity):
# MEDIAN_ISSUE_BATCH_SIZE = 10000
# @tf.function
# def median_per_column(x):
#   # split columns into batches because a GPU l2 cache issue
#   # (https://github.com/tensorflow/tensorflow/issues/43447)
#   # otherwise could have used tfp.stats.percentile(x, q=50, axis=0)
#   dataset = tf.data.Dataset.from_tensor_slices(tf.transpose(x)).batch(MEDIAN_ISSUE_BATCH_SIZE).map(
#       lambda batch: tfp.stats.percentile(batch, q=50, axis=1))
#
#   # annoyingly tf graph mode requires the following
#   # in order to extract a tensor from the dataset
#   # + it requires int64 converting to int32 in the TensorArray API
#   tensor_array = tf.TensorArray(dtype=dataset.element_spec.dtype,
#                                 size=tf.cast(tf.data.experimental.cardinality(dataset), tf.int32),
#                                 element_shape=dataset.element_spec.shape)
#   for i, t in dataset.enumerate():
#       tensor_array = tensor_array.write(tf.cast(i, tf.int32), t)
#   return tensor_array.concat()


class CountSketchCompression:
  def __init__(self, src_size, num_cols, num_rows,
               seed=(4, 2)):
    """
    Args:
        src_size: the number of elements in the vector that maps to the sketch
        num_cols: the number of columns (buckets) in the sketch
        num_rows: the number of rows in the sketch
        seed: a tuple with 2 numbers needed for reproducibility
    """

    self._num_cols = num_cols
    self._num_rows = num_rows

    h1, h2, h3, h4, h5, h6 = tf.unstack(
      tf.random.stateless_uniform(shape=(6, num_rows, 1),
                                  seed=seed,
                                  minval=0, maxval=LARGE_PRIME,
                                  dtype=tf.int64)
    )

    # [d]-indices as a row vector
    indices = tf.expand_dims(tf.range(src_size, dtype=tf.int64), axis=0)

    # computing bucket hashes (2-wise independence) (maps [src_size] to [num_cols])
    self._buckets = ((h1 * indices) + h2) % LARGE_PRIME % tf.cast(num_cols, dtype=tf.int64)

    # computing sign hashes (4-wise independence) (maps [src_size] to {-1, 1})
    self._signs = tf.cast(
      ((((h3 * indices + h4) * indices + h5) * indices + h6) % LARGE_PRIME % 2) * 2 - 1,
      dtype=tf.float32
    )

  def compress(self, tensor):
    # reshape needed otherwise second dimension is None sometimes...
    return tf.reshape(tf.math.bincount(
      self._buckets,
      weights=self._signs * tensor,
      minlength=self._num_cols,
      axis=-1
    ), shape=(self._num_rows, self._num_cols))

    # Before tensorflow 2.4 we had to use the following (left for posterity):
    # return tf.reshape(
    #     tf.map_fn(fn=lambda r: tf.math.bincount(
    #         tf.cast(self._buckets[r, :], tf.int32),
    #         weights=self._signs[r, :] * tensor,
    #         minlength=self._num_cols
    #     ), elems=tf.range(self._num_rows), dtype=tf.float32),
    #     shape=(self._num_rows, self._num_cols)
    # )

  def decompress(self, compressed_tensor):
    x = tf.gather(compressed_tensor, self._buckets, batch_dims=1) * self._signs
    return tfp.stats.percentile(x, q=50, axis=0)  # median per column
