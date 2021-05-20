from tensorflow_model_optimization.python.core.internal import tensor_encoding as te


def kashin_onebit_encoder():
  return te.core.EncoderComposer(
    te.stages.BitpackingEncodingStage(1)
  ).add_parent(
    te.stages.UniformQuantizationEncodingStage(1),
    te.stages.UniformQuantizationEncodingStage.ENCODED_VALUES_KEY
  ).add_parent(
    te.stages.research.KashinHadamardEncodingStage(),
    te.stages.research.KashinHadamardEncodingStage.ENCODED_VALUES_KEY
  ).add_parent(
    te.stages.FlattenEncodingStage(),
    te.stages.FlattenEncodingStage.ENCODED_VALUES_KEY
  ).make()
