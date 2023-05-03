"""Run masked LM/next sentence masked_lm pre-training for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import absl
import modeling
import optimization
import distribution_utils
import mlp_logging as mllog

from mlperf_logging.mllog import constants as mllog_constants

import tensorflow as tf
import horovod.tensorflow as hvd

flags = absl.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "input_file", None,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

flags.DEFINE_integer(
    "max_predictions_per_seq", 20,
    "Maximum number of masked LM predictions per sequence. "
    "Must match data generation.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_enum("optimizer", "adamw", ["adamw", "lamb"],
                  "The optimizer for training.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("poly_power", 1.0, "The power of poly decay.")

flags.DEFINE_integer("num_train_steps", 100000, "Number of training steps.")

flags.DEFINE_integer("num_warmup_steps", 10000, "Number of warmup steps.")

flags.DEFINE_integer("start_warmup_step", 0, "The starting step of warmup.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("max_eval_steps", 100, "Maximum number of eval steps.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_integer(
    "num_gpus", 0,
    "Use the GPU backend if this value is set to more than zero.")

flags.DEFINE_integer("steps_per_update", 1,
                     "The number of steps for accumulating gradients.")

flags.DEFINE_integer("keep_checkpoint_max", 5,
                     "The maximum number of checkpoints to keep.")

def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, optimizer, poly_power,
                     start_warmup_step):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """
    The `model_fn` for TPUEstimator.
    Returns an OutputSpec
    """

    tf.logging.info("IN MODEL_FN!!!")
    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    masked_lm_positions = features["masked_lm_positions"]
    masked_lm_ids = features["masked_lm_ids"]
    masked_lm_weights = features["masked_lm_weights"]
    next_sentence_labels = features["next_sentence_labels"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # Is model function describing a single step here?
    # DOes that mean we recrate the model each time??? Implausible
    (masked_lm_loss,
     masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
         bert_config, model.get_sequence_output(), model.get_embedding_table(),
         masked_lm_positions, masked_lm_ids, masked_lm_weights)

    (next_sentence_loss, next_sentence_example_loss,
     next_sentence_log_probs) = get_next_sentence_output(
         bert_config, model.get_pooled_output(), next_sentence_labels)

    total_loss = masked_lm_loss + next_sentence_loss

    tvars = tf.trainable_variables()

    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)

      tvar_index = {var.name.replace(":0", ""): var for var in tvars}
      assignment_map = collections.OrderedDict([
          (name, tvar_index.get(name, value))
          for name, value in assignment_map.items()
      ])

      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    # tf.logging.info("**** Trainable Variables ****")
    # for var in tvars:
    #   init_string = ""
    #   if var.name in initialized_variable_names:
    #     init_string = ", *INIT_FROM_CKPT*"
    #   tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
    #                   init_string)

    output_spec = None
    # It may be that we calculate the first loss (on the initial model) to create the graph flow
    # necessary for optimization
    if mode == tf.estimator.ModeKeys.TRAIN:
      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps,
          use_tpu, optimizer, poly_power, start_warmup_step, FLAGS.steps_per_update)

      if use_tpu:
        output_spec = tf.estimator.tpu.TPUEstimatorSpec(
            mode=mode,
            loss=total_loss,
            train_op=train_op,
            scaffold_fn=scaffold_fn)
      else:
        output_spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=total_loss,
            train_op=train_op)
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
                    masked_lm_weights, next_sentence_example_loss,
                    next_sentence_log_probs, next_sentence_labels):
        """Computes the loss and accuracy of the model."""
        masked_lm_log_probs = tf.reshape(masked_lm_log_probs,
                                         [-1, masked_lm_log_probs.shape[-1]])
        masked_lm_predictions = tf.argmax(
            masked_lm_log_probs, axis=-1, output_type=tf.int32)
        masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
        masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
        masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
        masked_lm_accuracy = tf.metrics.accuracy(
            labels=masked_lm_ids,
            predictions=masked_lm_predictions,
            weights=masked_lm_weights)
        masked_lm_mean_loss = tf.metrics.mean(
            values=masked_lm_example_loss, weights=masked_lm_weights)

        next_sentence_log_probs = tf.reshape(
            next_sentence_log_probs, [-1, next_sentence_log_probs.shape[-1]])
        next_sentence_predictions = tf.argmax(
            next_sentence_log_probs, axis=-1, output_type=tf.int32)
        next_sentence_labels = tf.reshape(next_sentence_labels, [-1])
        next_sentence_accuracy = tf.metrics.accuracy(
            labels=next_sentence_labels, predictions=next_sentence_predictions)
        next_sentence_mean_loss = tf.metrics.mean(
            values=next_sentence_example_loss)

        return {
            "masked_lm_accuracy": masked_lm_accuracy,
            "masked_lm_loss": masked_lm_mean_loss,
            "next_sentence_accuracy": next_sentence_accuracy,
            "next_sentence_loss": next_sentence_mean_loss,
        }

      eval_metrics = (metric_fn, [
          masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
          masked_lm_weights, next_sentence_example_loss,
          next_sentence_log_probs, next_sentence_labels
      ])
      if use_tpu:
        output_spec = tf.estimator.tpu.TPUEstimatorSpec(
            mode=mode,
            loss=total_loss,
            eval_metrics=eval_metrics,
            scaffold_fn=scaffold_fn)
      else:
        output_spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=total_loss,
            eval_metric_ops=metric_fn(
              masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
              masked_lm_weights, next_sentence_example_loss,
              next_sentence_log_probs, next_sentence_labels))

    else:
      raise ValueError("Only TRAIN and EVAL modes are supported: %s" % (mode))

    tf.logging.info("**** Returning OUTPUT SPEC ****")
    return output_spec

  return model_fn


def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
  """Get loss and log probs for the masked LM."""
  input_tensor = gather_indexes(input_tensor, positions)

  with tf.variable_scope("cls/predictions"):
    # We apply one more non-linear transformation before the output layer.
    # This matrix is not used after pre-training.
    with tf.variable_scope("transform"):
      input_tensor = tf.layers.dense(
          input_tensor,
          units=bert_config.hidden_size,
          activation=modeling.get_activation(bert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              bert_config.initializer_range))
      input_tensor = modeling.layer_norm(input_tensor)

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    output_bias = tf.get_variable(
        "output_bias",
        shape=[bert_config.vocab_size],
        initializer=tf.zeros_initializer())
    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    tf.logging.info(f'log_probs shape: {log_probs.get_shape()}')

    label_ids = tf.reshape(label_ids, [-1])
    label_weights = tf.reshape(label_weights, [-1])

    one_hot_labels = tf.one_hot(
        label_ids, depth=bert_config.vocab_size, dtype=tf.float32)
    tf.logging.info(f'one_hot_labels shape: {one_hot_labels.get_shape()}')

    # The `positions` tensor might be zero-padded (if the sequence is too
    # short to have the maximum number of predictions). The `label_weights`
    # tensor has a value of 1.0 for every real prediction and 0.0 for the
    # padding predictions.
    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
    numerator = tf.reduce_sum(label_weights * per_example_loss)
    denominator = tf.reduce_sum(label_weights) + 1e-5
    loss = numerator / denominator

  return (loss, per_example_loss, log_probs)


def get_next_sentence_output(bert_config, input_tensor, labels):
  """Get loss and log probs for the next sentence prediction."""

  # Simple binary classification. Note that 0 is "next sentence" and 1 is
  # "random sentence". This weight matrix is not used after pre-training.
  with tf.variable_scope("cls/seq_relationship"):
    output_weights = tf.get_variable(
        "output_weights",
        shape=[2, bert_config.hidden_size],
        initializer=modeling.create_initializer(bert_config.initializer_range))
    output_bias = tf.get_variable(
        "output_bias", shape=[2], initializer=tf.zeros_initializer())

    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    labels = tf.reshape(labels, [-1])
    one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    return (loss, per_example_loss, log_probs)


def gather_indexes(sequence_tensor, positions):
  """Gathers the vectors at the specific positions over a minibatch."""
  sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
  batch_size = sequence_shape[0]
  seq_length = sequence_shape[1]
  width = sequence_shape[2]

  flat_offsets = tf.reshape(
      tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
  flat_positions = tf.reshape(positions + flat_offsets, [-1])
  flat_sequence_tensor = tf.reshape(sequence_tensor,
                                    [batch_size * seq_length, width])
  output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
  return output_tensor


def input_fn_builder(input_files,
                     batch_size,
                     max_seq_length,
                     max_predictions_per_seq,
                     is_training,
                     input_context = None,
                     num_cpu_threads=4,
                     num_eval_steps=1):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  def input_fn(params, input_context = None):
    """The actual input function."""
    # This is the per GPU batch size
    batch_size = params["batch_size"]

    name_to_features = {
        "input_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "masked_lm_positions":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_ids":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_weights":
            tf.FixedLenFeature([max_predictions_per_seq], tf.float32),
        "next_sentence_labels":
            tf.FixedLenFeature([1], tf.int64),
    }

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    if is_training: 
      d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
      # An input context will be passed when input_fn is called during training.
      if input_context:
        tf.logging.info(
            'Sharding the dataset: input_pipeline_id=%d num_input_pipelines=%d' % (
            input_context.input_pipeline_id, input_context.num_input_pipelines))
        tf.logging.info(f"Input context: {str(input_context)}")
        # d = d.shard(input_context.num_input_pipelines,
        #             input_context.input_pipeline_id)

      # Horovod, shard the dataset between workers
      d = d.shard(hvd.size(), hvd.rank())
      d = d.shuffle(buffer_size=len(input_files))

      # `cycle_length` is the number of parallel files that get read.
      # Minimum btw number of cpu threads we want to use or number of files
      # cycle_length = min(num_cpu_threads, len(input_files))
      cycle_length = 8

      tf.logging.info('parallel interleave cycle_length=%d' % (cycle_length))
      # Here we actually create the dataset using the filenames stored in d
      # "it gets elements from cycle_length nested datasets in parallel, which increases the throughput, especially in the presence of stragglers."

      # `sloppy` mode means that the interleaving is not exact. This adds
      # even more randomness to the training pipeline.
      # Block length is 1 by default = The number of consecutive elements to pull from an input Dataset before advancing to the next input Dataset.
      # So it will produce a dataset of TFRecords,  pick 1 file from each file in sequence
      # https://www.tensorflow.org/api_docs/python/tf/data/experimental/parallel_interleave

      # buffer_output_elements	The number of elements each iterator being interleaved should buffer 
      # (similar to the .prefetch() transformation for each interleaved iterator). is None by default!
      # parallel_interleave() maps map_func across its input to produce nested datasets, and outputs their elements interleaved. 
      # Unlike tf.data.Dataset.interleave, it gets elements from cycle_length nested datasets in parallel, 
      # which increases the throughput, especially in the presence of stragglers. 
      # Furthermore, the sloppy argument can be used to improve performance, by relaxing the requirement that the outputs 
      # are produced in a deterministic order, and allowing the implementation to skip over nested datasets whose elements are not readily available when requested.
      d = d.apply(
          tf.data.experimental.parallel_interleave(
              tf.data.TFRecordDataset,
              sloppy=is_training,
              cycle_length=cycle_length))

      # Create a buffer that will randomly pick from 1000 TFRecords 
      # The 1001'th element can never be returned first
      d = d.shuffle(buffer_size=1000)
      # Dataset is set to repeat

      # Makes the dataset become effectvely infinite
      # When do we load new files then?
      d = d.repeat()
    else:
      d = tf.data.TFRecordDataset(input_files)
      d = d.take(batch_size * num_eval_steps)
      # Since we evaluate for a fixed number of steps we don't want to encounter
      # out-of-range exceptions.
      d = d.repeat()

    # We must `drop_remainder` on training because the TPU requires fixed
    # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
    # and we *don't* want to drop the remainder, otherwise we wont cover
    # every sample.
    # https://www.tensorflow.org/api_docs/python/tf/data/experimental/map_and_batch
    d = d.apply(
        # batch_size * num_parallel_batches elements will be processed in parallel.
        tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            num_parallel_batches=num_cpu_threads,
            drop_remainder=True))
    
    return d

  return input_fn


def _decode_record(record, name_to_features):
  """Decodes a record to a TensorFlow example."""
  tf.logging.info("Decoding record")
  example = tf.parse_single_example(record, name_to_features)

  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # So cast all int64 to int32.
  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
      t = tf.to_int32(t)
    example[name] = t

  return example


class CheckpointHook(tf.train.CheckpointSaverHook):
  """Add MLPerf logging to checkpoint saving."""

  def __init__(self, num_train_steps, *args, **kwargs):
    super(CheckpointHook, self).__init__(*args, **kwargs)
    self.num_train_steps = num_train_steps
    self.previous_step = None

  def _save(self, session, step):
    if self.previous_step is not None:
      mllog.mllog_end(key=mllog_constants.BLOCK_STOP,
                      metadata={"first_step_num": self.previous_step + 1,
                          "step_count": step - self.previous_step})
    else:
      # First time this gets called
      mllog.mllog_end(key=mllog_constants.INIT_STOP)

    self.previous_step = step
    # Don't checkpoint right at the start
    mllog.mllog_start(key="checkpoint_start", metadata={"step_num" : step}) 
    return_value = super(CheckpointHook, self)._save(session, step)
    mllog.mllog_end(key="checkpoint_stop", metadata={"step_num" : step})

    if step < self.num_train_steps:
        mllog.mllog_start(key=mllog_constants.BLOCK_START, metadata={"first_step_num": step + 1})

    return return_value


def main(_):

  tf.logging.set_verbosity(tf.logging.INFO)
  
  # Initialize Horovod
  hvd.init()

  if not FLAGS.do_train and not FLAGS.do_eval:
    raise ValueError("At least one of `do_train` or `do_eval` must be True.")

  tf.gfile.MakeDirs(FLAGS.output_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)

  # mlperf_logger = mllog.get_mlperf_logger(FLAGS.output_dir, 'bert.log')
  if FLAGS.do_train:
    mllog.mllog_start(key=mllog_constants.INIT_START)

  # bert_config.json parametrized the BERT large model in the paper
  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)


  input_files = []

  # This returns the file names matching the given glob pattern
  # Aka. will fetch th enames of all input tfrecords
  for input_pattern in FLAGS.input_file.split(","):
    input_files.extend(tf.gfile.Glob(input_pattern))

  # tf.logging.info("*** Input Files ***")
  # for input_file in input_files:
  #   tf.logging.info("  %s" % input_file)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  # Init
  model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,    # /wiki/ckpt/model.ckpt-28252
      learning_rate=FLAGS.learning_rate,        # 0.0001
      num_train_steps=FLAGS.num_train_steps,    # 107538
      num_warmup_steps=FLAGS.num_warmup_steps,  # 0
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu, 
      optimizer=FLAGS.optimizer,                # lamb
      poly_power=FLAGS.poly_power,
      start_warmup_step=FLAGS.start_warmup_step 
    )

  if FLAGS.use_tpu:
    is_per_host = tf.estimator.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.estimator.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        keep_checkpoint_max=FLAGS.keep_checkpoint_max,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.estimator.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.estimator.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size)
  else:
    # GPU path uses MirroredStrategy.

    # Creates session config. allow_soft_placement = True, is required for
    # multi-GPU and is not harmful for other modes.
    # See https://github.com/tensorflow/tensorflow/blob/v1.15.0/tensorflow/core/protobuf/config.proto#L360
    session_config = tf.compat.v1.ConfigProto(
        inter_op_parallelism_threads=8,
        allow_soft_placement=True)

    # Pin GPU to be used to process local rank (one GPU per process)
    session_config.gpu_options.visible_device_list = str(hvd.local_rank())

    # # TODO: What is num_packs?
    # # We could use a DGX-1 optimized version of all_reduce_alg here: 
    # distribution_strategy = distribution_utils.get_distribution_strategy(
    #     distribution_strategy="mirrored",
    #     num_gpus=FLAGS.num_gpus,
    #     all_reduce_alg="nccl",
    #     num_packs=0)

    # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
    # Need to be the same between confi and estimator
    model_dir = FLAGS.output_dir

    # Make an estimator run configuration using the distribution strategy and session config
    dist_gpu_config = tf.estimator.RunConfig(
        # train_distribute=distribution_strategy,
        model_dir=model_dir,
        session_config=session_config,
        keep_checkpoint_max=FLAGS.keep_checkpoint_max,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
    )

    # Estimator is configured with the dsitributed GPU config that 
    # implements the distribution strategy, and the global batch size
    hparams = {"batch_size": FLAGS.train_batch_size}
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=dist_gpu_config,
        model_dir=model_dir,
        params=hparams,
    )

  if FLAGS.do_train:
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Overall batch size = %d", FLAGS.train_batch_size)
    batch_size = FLAGS.train_batch_size
    if FLAGS.num_gpus > 1:
      tf.logging.info("  Using %d GPUs", FLAGS.num_gpus)
      batch_size = distribution_utils.per_replica_batch_size(
            batch_size, FLAGS.num_gpus)
    hparams = {"batch_size": batch_size}
    tf.logging.info("  Per GPU batch size = %d", batch_size)

    # The train input function is given the per gpu batch size
    # and has a configurable number of CPU threads, presumably to load data from disk
    train_input_fn = input_fn_builder(
        input_files=input_files,
        batch_size=batch_size,    # not actually used, input_fn will use the batch size in hparams when it's called, which is equal
        max_seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        is_training=True,
        input_context=None, # Always None so the dataset is not sharded? No, the distribution strategy passes the input context to the input_fn
        num_cpu_threads=8)

    checkpoint_hook = CheckpointHook(
        num_train_steps=FLAGS.num_train_steps,
        checkpoint_dir=model_dir,
        save_steps=FLAGS.save_checkpoints_steps)
    
    # profiler_hook = tf.estimator.ProfilerHook(
    #       save_steps=10,
    #       output_dir=FLAGS.log_dir,
    #       show_dataflow=True,
    #       show_memory=False
    #     )

    # Horovod: BroadcastGlobalVariablesHook broadcasts initial variable states from
    # rank 0 to all other processes. This is necessary to ensure consistent
    # initialization of all workers when training is started with random weights or
    # restored from a checkpoint.
    bcast_hook = hvd.BroadcastGlobalVariablesHook(0)

    hooks = [checkpoint_hook, bcast_hook] if hvd.rank() == 0 else [bcast_hook]

    mllog.mlperf_submission_log()
    mllog.mlperf_run_param_log()
    
    mllog.mllog_start(key=mllog_constants.RUN_START)
    if FLAGS.use_tpu:
      estimator.train(input_fn=train_input_fn, max_steps=FLAGS.num_train_steps,
          hooks=hooks)
    else:
      tf.logging.info("********** CALLING ESTIMATOR.TRAIN() **************")
      estimator.train(input_fn=lambda input_context=None: train_input_fn(
          params=hparams, input_context=input_context), max_steps=FLAGS.num_train_steps,
          hooks=hooks)
    mllog.mllog_end(key=mllog_constants.RUN_STOP)

  if FLAGS.do_eval:
    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
    batch_size = FLAGS.eval_batch_size
    if FLAGS.num_gpus > 1:
      batch_size = distribution_utils.per_replica_batch_size(
            batch_size, FLAGS.num_gpus)
    hparams = {"batch_size": batch_size}
    eval_input_fn = input_fn_builder(
        input_files=input_files,
        batch_size=batch_size,
        max_seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        is_training=False,
        input_context=None,
        num_cpu_threads=8,
        num_eval_steps=FLAGS.max_eval_steps)

    mllog.mllog_start(key=mllog_constants.EVAL_START)
    result = estimator.evaluate(
        input_fn=lambda input_context=None: eval_input_fn(
            params=hparams, input_context=input_context),
        steps=FLAGS.max_eval_steps)
        
    global_step = result["global_step"]
    mllog.mllog_end(key=mllog_constants.EVAL_STOP, value=global_step,
                      metadata={"step_num": global_step})
    mllog.mllog_event(key=mllog_constants.EVAL_ACCURACY,
                      value=result["masked_lm_accuracy"],
                      metadata={"step_num": global_step})

    output_eval_file = os.path.join(FLAGS.log_dir, "eval_results.txt")
    with tf.gfile.GFile(output_eval_file, "w") as writer:
      tf.logging.info("***** Eval results *****")
      for key in sorted(result.keys()):
        tf.logging.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))



if __name__ == "__main__":
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")

  absl.app.run(main)
