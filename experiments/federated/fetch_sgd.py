from collections import namedtuple

import tensorflow as tf
from sympy import nextprime

from experiments.federated.aggregation_process import GradientAggregationSpec, AggregationSpec, identity_aggr
from experiments.federated.count_sketch import CountSketchCompression


SharedState = namedtuple('FetchSgdSharedState', ['momentum_coef', 'learning_rate', 'k'])
PerVarState = namedtuple('FetchSgdPerVarState', ['momentum', 'error'])


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

def build_fetch_sgd(var_types, var_names,
                    num_rows, num_cols, momentum, lr, k, require_size_larger_than, **kwargs) -> AggregationSpec:

    # for comparing with DRIVE we are not using num_cols as a common param
    # instead we calculate a sketch that uses 1 bit
    # and round it to the next prime (required by the count-sketch data structure)
    def per_grad_factoy(src_size):
        num_cols = nextprime(src_size // 32)

        def initialize(type, name):
            return PerVarState(
                momentum=tf.zeros(shape=[num_rows, num_cols], dtype=type.dtype, name=f'{name}/momentum'),
                error=tf.zeros(shape=[num_rows, num_cols], dtype=type.dtype, name=f'{name}/error'),
            )

        def encode(grad): # todo verify using size instead of shape works
            return CountSketchCompression(src_size, num_cols, num_rows).compress(tf.reshape(grad, [-1]))

        @tf.function
        def decode_post_aggregate(shared: SharedState, var_state: PerVarState, original_shape, encoded_grad):
            momentum = var_state.momentum
            error = var_state.error

            encoded_grad = momentum = shared.momentum_coef * momentum + encoded_grad
            encoded_grad = error + shared.learning_rate * encoded_grad

            # num_elements = tf.reduce_prod(original_shape)
            compression = CountSketchCompression(src_size,
                                                 num_cols,
                                                 num_rows)

            grad = compression.decompress(encoded_grad)
            if src_size > shared.k:
                grad = abs_top_k(grad, shared.k)

            # algorithm 1 line 14 would be 'error = encoded_grad - compression.compress(grad)'
            # but the paper says: "On line 14 of Algorithm 1, we zero out the nonzero coordinates of S(∆t) in Ste
            # instead of subtracting S(∆t). Empirically, doing so stabilizes the optimization."
            # error = encoded_grad - compression.compress(grad)
            error = tf.where(tf.equal(compression.compress(grad), 0), encoded_grad, 0)

            return tf.reshape(grad, original_shape), PerVarState(
                momentum=momentum,
                error=error
            )

        return GradientAggregationSpec(
            initialize=initialize,
            encode=encode,
            decode_post_aggregate=decode_post_aggregate,
        )

    gradient_aggregation_specs = [per_grad_factoy(type.shape.num_elements()) if type.shape.num_elements() > require_size_larger_than else identity_aggr for type, name in zip(var_types, var_names)]

    def initialize_shared() -> SharedState:
        return SharedState(
            momentum_coef=momentum, # in the paper 0.9 or 0
            # TODO "When we train for 1 epoch, the pivot epoch of the learning rate
            #  schedule is 0.2, and the peak learning rate is 0.01."
            learning_rate=lr, # 0.2
            k=k, # 10000,
        )

    return AggregationSpec(initialize_shared, gradient_aggregation_specs)
