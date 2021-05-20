from tensorflow_model_optimization.python.core.internal import tensor_encoding as te

from drive_tf.drive_encoding_stages import DriveSignStage, DrivePlusClusteringStage


def drive_encoder():
  return te.core.core_encoder.EncoderComposer(
    te.stages.BitpackingEncodingStage(1)
  ).add_parent(
    DriveSignStage(),
    DriveSignStage.ENCODED_VALUES_KEY
  ).add_parent(
    te.stages.HadamardEncodingStage(),
    te.stages.HadamardEncodingStage.ENCODED_VALUES_KEY
  ).add_parent(
    te.stages.FlattenEncodingStage(),
    te.stages.FlattenEncodingStage.ENCODED_VALUES_KEY
  ).make()


def drive_plus_encoder(clustering: str = 'floyd'):
  return te.core.core_encoder.EncoderComposer(
    te.stages.BitpackingEncodingStage(1)
  ).add_parent(
    DrivePlusClusteringStage(clustering),
    DrivePlusClusteringStage.ENCODED_VALUES_KEY
  ).add_parent(
    te.stages.HadamardEncodingStage(),
    te.stages.HadamardEncodingStage.ENCODED_VALUES_KEY
  ).add_parent(
    te.stages.FlattenEncodingStage(),
    te.stages.FlattenEncodingStage.ENCODED_VALUES_KEY
  ).make()
