from collections import namedtuple
from typing import Callable, Any

import tensorflow as tf
import tensorflow_federated as tff

State = namedtuple('AggregatorState', ['per_var', 'shared'])

GradientAggregationSpec = namedtuple('GradientAggregationSpec', ['initialize', 'encode', 'decode_post_aggregate'])
AggregationSpec = namedtuple('AggregationSpec', ['initialize_shared', 'gradient_aggregation_specs'])

identity_aggr = GradientAggregationSpec(
    initialize=lambda type, name: (),
    encode=lambda grad: tf.identity(grad),
    decode_post_aggregate=lambda shared, var_state, original_shape, encoded_grad: (tf.identity(encoded_grad), ())
)


def _get_model_metadata(model_fn: Callable[[], tff.learning.Model]):
    with tf.Graph().as_default():
        model = model_fn()
    var_types = tff.framework.type_from_tensors(model.trainable_variables)
    var_names = [v._shared_name for v in model.trainable_variables]
    return var_types, var_names


def build_aggregation_process(
        model_fn: Callable[[], tff.learning.Model],
        spec_fn: Callable[[Any, Any], AggregationSpec],
        **kwargs
        ) -> tff.templates.MeasuredProcess:
    var_types, var_names = _get_model_metadata(model_fn)

    spec = spec_fn(var_types, var_names, **kwargs)

    @tff.tf_computation
    def initialize_fn_comp():
        return State(
            per_var=[s.initialize(type, name) for s, type, name in zip(spec.gradient_aggregation_specs, var_types, var_names)],
            shared=spec.initialize_shared()
        )

    @tff.federated_computation()
    def initialize_fn():
        return tff.federated_eval(initialize_fn_comp, tff.SERVER)

    state_type = initialize_fn_comp.type_signature.result

    @tff.tf_computation(var_types)
    def client_encode_fn(model_delta):
        return [s.encode(-delta) for delta, s in zip(model_delta, spec.gradient_aggregation_specs)]

    encoded_grads_type = client_encode_fn.type_signature.result

    @tff.tf_computation(state_type, encoded_grads_type)
    def post_aggregate_fn(state, encoded_grads):
        grads, per_var = zip(*[_[0].decode_post_aggregate(state.shared, _[1], _[2], _[3]) for _ in
                               zip(spec.gradient_aggregation_specs, state.per_var, [v.shape.as_list() for v in var_types], encoded_grads)])
        deltas = [-_ for _ in grads] # tff expects deltas

        return State(
            per_var=per_var,
            shared=state.shared
        ), deltas

    @tff.federated_computation(
        tff.FederatedType(state_type, tff.SERVER),
        tff.FederatedType(var_types, tff.CLIENTS),
        tff.FederatedType(tf.float32, tff.CLIENTS))
    def next_fn(state, model_deltas, weight):
        # get the encoding parameters from the state
        encoded_grads = tff.federated_map(client_encode_fn, model_deltas)

        # aggregate
        aggregated_encoded_grads = tff.federated_mean(encoded_grads, weight)

        state, aggregated_deltas = tff.federated_map(post_aggregate_fn, (state, aggregated_encoded_grads))

        # send them to the clients
        return tff.templates.MeasuredProcessOutput(
            state=state,
            result=aggregated_deltas,
            measurements=tff.federated_value((), tff.SERVER)
        )

    return tff.templates.AggregationProcess(initialize_fn=initialize_fn, next_fn=next_fn)
