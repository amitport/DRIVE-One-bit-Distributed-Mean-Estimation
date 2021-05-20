import tensorflow as tf


@tf.function
def abs_top_k(tensor, k):  # k should be a tf.constant to avoid retracing
    orig_shape = tf.shape(tensor)

    x = tf.reshape(tensor, [-1])  # flatten
    _, indices = tf.math.top_k(tf.math.abs(x), k)  # absolute top-k indices
    x = tf.scatter_nd(
        indices=tf.expand_dims(indices, -1),  # scatter_nd expect this format
        updates=tf.gather(x, indices),  # find original non-absolute value
        shape=tf.shape(x))
    return tf.reshape(x, orig_shape)  # unflatten