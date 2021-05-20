import collections
import os.path
from typing import Callable

import tensorflow as tf
import tensorflow_federated as tff
from absl import app
from absl import flags
from tensorflow_model_optimization.python.core.internal import tensor_encoding as te

from experiments.federated.aggregation_process import build_aggregation_process
from experiments.federated.encoders import ENCODERS
from experiments.federated.fetch_sgd import build_fetch_sgd
from google_tff_research.optimization.cifar100 import federated_cifar100
from google_tff_research.optimization.emnist import federated_emnist
from google_tff_research.optimization.emnist_ae import federated_emnist_ae
from google_tff_research.optimization.shakespeare import federated_shakespeare
from google_tff_research.optimization.shared import optimizer_utils
from google_tff_research.optimization.shared import training_specs
from google_tff_research.optimization.stackoverflow import federated_stackoverflow
from google_tff_research.optimization.stackoverflow_lr import federated_stackoverflow_lr
from google_tff_research.utils import training_loop
from google_tff_research.utils import utils_impl

_SUPPORTED_TASKS = [
  'cifar100', 'emnist_cr', 'shakespeare', 'stackoverflow_nwp',
]

_SUPPORTED_AGGR = [
                    'mean', 'fetch_sgd',
                  ] + list(ENCODERS.keys())

with utils_impl.record_hparam_flags() as optimizer_flags:
  # Defining optimizer flags
  optimizer_utils.define_optimizer_flags('client')
  optimizer_utils.define_optimizer_flags('server')

with utils_impl.record_hparam_flags() as shared_flags:
  # Federated training hyperparameters
  flags.DEFINE_integer('client_epochs_per_round', 1,
                       'Number of epochs in the client to take per round.')
  flags.DEFINE_integer('client_batch_size', 20, 'Batch size on the clients.')
  flags.DEFINE_integer('clients_per_round', 10,
                       'How many clients to sample per round.')
  flags.DEFINE_integer('client_datasets_random_seed', 1,
                       'Random seed for client sampling.')

  # Training loop configuration
  flags.DEFINE_string(
    'experiment_name', None, 'The name of this experiment. Will be append to '
                             '--root_output_dir to separate experiment results.')
  flags.mark_flag_as_required('experiment_name')
  flags.DEFINE_string('root_output_dir', '/tmp/fed_opt/',
                      'Root directory for writing experiment output.')
  flags.DEFINE_integer('total_rounds', 200, 'Number of total training rounds.')
  flags.DEFINE_integer(
    'rounds_per_eval', 1,
    'How often to evaluate the global model on the validation dataset.')
  flags.DEFINE_integer('rounds_per_checkpoint', 50,
                       'How often to checkpoint the global model.')

with utils_impl.record_hparam_flags() as task_flags:
  # Task specification
  flags.DEFINE_enum('task', None, _SUPPORTED_TASKS,
                    'Which task to perform federated training on.')

with utils_impl.record_hparam_flags() as cifar100_flags:
  # CIFAR-100 flags
  flags.DEFINE_integer('cifar100_crop_size', 24, 'The height and width of '
                                                 'images after preprocessing.')
  flags.DEFINE_bool(
    'cifar100_distort_train_images', True, 'If set to True, '
                                           'train images will be randomly cropped. Otherwise, all '
                                           'images will simply be resized.')

with utils_impl.record_hparam_flags() as emnist_cr_flags:
  # EMNIST CR flags
  flags.DEFINE_enum(
    'emnist_cr_model', 'cnn', ['cnn', '2nn'], 'Which model to '
                                              'use. This can be a convolutional model (cnn) or a two '
                                              'hidden-layer densely connected network (2nn).')

with utils_impl.record_hparam_flags() as shakespeare_flags:
  # Shakespeare flags
  flags.DEFINE_integer(
    'shakespeare_sequence_length', 80,
    'Length of character sequences to use for the RNN model.')

with utils_impl.record_hparam_flags() as so_nwp_flags:
  # Stack Overflow NWP flags
  flags.DEFINE_integer('so_nwp_vocab_size', 10000, 'Size of vocab to use.')
  flags.DEFINE_integer('so_nwp_num_oov_buckets', 1,
                       'Number of out of vocabulary buckets.')
  flags.DEFINE_integer('so_nwp_sequence_length', 20,
                       'Max sequence length to use.')
  flags.DEFINE_integer('so_nwp_max_elements_per_user', 1000, 'Max number of '
                                                             'training sentences to use per user.')
  flags.DEFINE_integer(
    'so_nwp_num_validation_examples', 10000, 'Number of examples '
                                             'to use from test set for per-round validation.')

with utils_impl.record_hparam_flags() as so_lr_flags:
  # Stack Overflow LR flags
  flags.DEFINE_integer('so_lr_vocab_tokens_size', 10000,
                       'Vocab tokens size used.')
  flags.DEFINE_integer('so_lr_vocab_tags_size', 500, 'Vocab tags size used.')
  flags.DEFINE_integer(
    'so_lr_num_validation_examples', 10000, 'Number of examples '
                                            'to use from test set for per-round validation.')
  flags.DEFINE_integer('so_lr_max_elements_per_user', 1000,
                       'Max number of training '
                       'sentences to use per user.')

with utils_impl.record_hparam_flags() as aggr_flags:
  # Task specification
  flags.DEFINE_enum('aggr', 'mean', _SUPPORTED_AGGR, 'Which aggregation process to use')

with utils_impl.record_hparam_flags() as encoder_flags:
  flags.DEFINE_integer('encoder_require_size_larger_than', 10000,
                       'Values with same or fewer elements will not be encoded')
  flags.DEFINE_boolean('encoder_require_floating', True,
                       'Whether to only encode floating variable')

with utils_impl.record_hparam_flags() as fetch_sgd_flags:
  # Task specification
  flags.DEFINE_integer('fetch_sgd_num_rows', 5, 'The number of rows in the sketch')
  flags.DEFINE_integer('fetch_sgd_num_cols', 10000, 'The number of columns in the sketch')
  flags.DEFINE_float('fetch_sgd_momentum', 0., 'momentum applied to the aggregated sketched gradient (rho)')
  flags.DEFINE_float('fetch_sgd_lr', 1., 'learning rate applied to the aggregated sketched gradient (eta)')
  flags.DEFINE_integer('fetch_sgd_k', 10000, 'The number of coordinates to take after unsketching (k)')

with utils_impl.record_hparam_flags() as hadamard_quantization_flags:
  flags.DEFINE_integer('hadamard_quantization_bits', 1, 'Number of bits to quantize into')

FLAGS = flags.FLAGS

TASK_FLAGS = collections.OrderedDict(
  cifar100=cifar100_flags,
  emnist_cr=emnist_cr_flags,
  shakespeare=shakespeare_flags,
  stackoverflow_nwp=so_nwp_flags,
  stackoverflow_lr=so_lr_flags)

TASK_FLAG_PREFIXES = collections.OrderedDict(
  cifar100='cifar100',
  emnist_cr='emnist_cr',
  emnist_ae='emnist_ae',
  shakespeare='shakespeare',
  stackoverflow_nwp='so_nwp',
  stackoverflow_lr='so_lr')

AGGR_FLAGS = collections.OrderedDict(
  fetch_sgd=fetch_sgd_flags,
  hadamard_quantization=hadamard_quantization_flags,
)

AGGR_FLAG_PREFIXES = dict(zip(_SUPPORTED_AGGR, _SUPPORTED_AGGR))


def _write_hparam_flags():
  """Creates an ordered dictionary of hyperparameter flags and writes to CSV."""
  hparam_dict = utils_impl.lookup_flag_values(shared_flags)

  # Update with optimizer flags corresponding to the chosen optimizers.
  opt_flag_dict = utils_impl.lookup_flag_values(optimizer_flags)
  opt_flag_dict = optimizer_utils.remove_unused_flags('client', opt_flag_dict)
  opt_flag_dict = optimizer_utils.remove_unused_flags('server', opt_flag_dict)
  hparam_dict.update(opt_flag_dict)

  # Update with task-specific flags.
  task_name = FLAGS.task
  if task_name in TASK_FLAGS:
    task_hparam_dict = utils_impl.lookup_flag_values(TASK_FLAGS[task_name])
    hparam_dict.update(task_hparam_dict)
  hparam_dict.update([('task', task_name)])

  aggr_name = FLAGS.aggr
  if aggr_name in AGGR_FLAGS:
    aggr_hparam_dict = utils_impl.lookup_flag_values(AGGR_FLAGS[aggr_name])
    hparam_dict.update(aggr_hparam_dict)
  hparam_dict.update([('aggr', aggr_name)])

  hparam_dict.update(utils_impl.lookup_flag_values(encoder_flags))

  results_dir = os.path.join(FLAGS.root_output_dir, 'results',
                             FLAGS.experiment_name)
  utils_impl.create_directory_if_not_exists(results_dir)
  hparam_file = os.path.join(results_dir, 'hparams.csv')
  utils_impl.atomic_write_series_to_csv(hparam_dict, hparam_file)


def _get_aggr_args():
  aggr_name = FLAGS.aggr
  aggr_args = collections.OrderedDict()

  if aggr_name in AGGR_FLAGS:
    aggr_flag_list = AGGR_FLAGS[aggr_name]
    aggr_flag_dict = utils_impl.lookup_flag_values(aggr_flag_list)
    aggr_flag_prefix = AGGR_FLAG_PREFIXES[aggr_name]
    for (key, value) in aggr_flag_dict.items():
      if key.startswith(aggr_flag_prefix):
        key = key[len(aggr_flag_prefix):].lstrip('_-')
      aggr_args[key] = value

  # drop 'encoder_' prefix
  aggr_args.update([(key[8:], value) for (key, value) in utils_impl.lookup_flag_values(encoder_flags).items()])

  return aggr_args


def build_sum_encoder_fn(encoder,
                         # 10000 is enough to make sure we don't encode normalization or last layers
                         # (and the same is used in google-research/federated)
                         require_size_larger_than=10000,
                         require_floating=True,
                         **kwargs):
  def encoder_fn(spec):
    if spec.shape.num_elements() > require_size_larger_than:
      return te.encoders.as_gather_encoder(
        encoder(**kwargs),
        spec)
    else:
      return te.encoders.as_gather_encoder(te.encoders.identity(), spec)

  return encoder_fn


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Expected no command-line arguments, '
                         'got: {}'.format(argv))

  client_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('client')
  server_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('server')

  aggr_args = _get_aggr_args()

  def iterative_process_builder(
          model_fn: Callable[[], tff.learning.Model],
  ) -> tff.templates.IterativeProcess:
    """Creates an iterative process using a given TFF `model_fn`.

    Args:
      model_fn: A no-arg function returning a `tff.learning.Model`.

    Returns:
      A `tff.templates.IterativeProcess`.
    """

    if FLAGS.task == 'shakespeare' or FLAGS.task == 'stackoverflow_nwp':

      def client_weight_fn(local_outputs):
        return tf.cast(tf.squeeze(local_outputs['num_tokens']), tf.float32)
    else:
      client_weight_fn = None

    if FLAGS.aggr == 'mean':
      aggregation_factory = None
      aggregation_process = None
    elif FLAGS.aggr == 'fetch_sgd':
      aggregation_factory = None
      aggregation_process = build_aggregation_process(model_fn, build_fetch_sgd, **aggr_args)
    else:
      aggregation_factory = tff.aggregators.MeanFactory(
        tff.aggregators.EncodedSumFactory(build_sum_encoder_fn(ENCODERS[FLAGS.aggr], **aggr_args)))
      aggregation_process = None

    return tff.learning.build_federated_averaging_process(
      model_fn=model_fn,
      client_optimizer_fn=client_optimizer_fn,
      server_optimizer_fn=server_optimizer_fn,
      aggregation_process=aggregation_process,
      model_update_aggregation_factory=aggregation_factory,
      client_weighting=client_weight_fn,
    )

  task_spec = training_specs.TaskSpec(
    iterative_process_builder=iterative_process_builder,
    client_epochs_per_round=FLAGS.client_epochs_per_round,
    client_batch_size=FLAGS.client_batch_size,
    clients_per_round=FLAGS.clients_per_round,
    client_datasets_random_seed=FLAGS.client_datasets_random_seed)

  if FLAGS.task == 'cifar100':
    runner_spec = federated_cifar100.configure_training(
      task_spec,
      crop_size=FLAGS.cifar100_crop_size,
      distort_train_images=FLAGS.cifar100_distort_train_images)
  elif FLAGS.task == 'emnist_cr':
    runner_spec = federated_emnist.configure_training(
      task_spec, model=FLAGS.emnist_cr_model)
  elif FLAGS.task == 'emnist_ae':
    runner_spec = federated_emnist_ae.configure_training(task_spec)
  elif FLAGS.task == 'shakespeare':
    runner_spec = federated_shakespeare.configure_training(
      task_spec, sequence_length=FLAGS.shakespeare_sequence_length)
  elif FLAGS.task == 'stackoverflow_nwp':
    runner_spec = federated_stackoverflow.configure_training(
      task_spec,
      vocab_size=FLAGS.so_nwp_vocab_size,
      num_oov_buckets=FLAGS.so_nwp_num_oov_buckets,
      sequence_length=FLAGS.so_nwp_sequence_length,
      max_elements_per_user=FLAGS.so_nwp_max_elements_per_user,
      num_validation_examples=FLAGS.so_nwp_num_validation_examples)
  elif FLAGS.task == 'stackoverflow_lr':
    runner_spec = federated_stackoverflow_lr.configure_training(
      task_spec,
      vocab_tokens_size=FLAGS.so_lr_vocab_tokens_size,
      vocab_tags_size=FLAGS.so_lr_vocab_tags_size,
      max_elements_per_user=FLAGS.so_lr_max_elements_per_user,
      num_validation_examples=FLAGS.so_lr_num_validation_examples)
  else:
    raise ValueError(
      '--task flag {} is not supported, must be one of {}.'.format(
        FLAGS.task, _SUPPORTED_TASKS))

  _write_hparam_flags()

  training_loop.run(
    iterative_process=runner_spec.iterative_process,
    client_datasets_fn=runner_spec.client_datasets_fn,
    validation_fn=runner_spec.validation_fn,
    test_fn=runner_spec.test_fn,
    total_rounds=FLAGS.total_rounds,
    experiment_name=FLAGS.experiment_name,
    root_output_dir=FLAGS.root_output_dir,
    rounds_per_eval=FLAGS.rounds_per_eval,
    rounds_per_checkpoint=FLAGS.rounds_per_checkpoint)


if __name__ == '__main__':
  app.run(main)
