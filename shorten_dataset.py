import os
from re import I
import absl

import collections
import tensorflow as tf

flags = absl.flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "input_file", None,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

tf.compat.v1.enable_eager_execution()

NAME_TO_FEATURES = {
    "input_ids":
        tf.io.FixedLenFeature([512], tf.int64),
    "input_mask":
        tf.io.FixedLenFeature([512], tf.int64),
    "segment_ids":
        tf.io.FixedLenFeature([512], tf.int64),
    "masked_lm_positions":
        tf.io.FixedLenFeature([76], tf.int64),
    "masked_lm_ids":
        tf.io.FixedLenFeature([76], tf.int64),
    "masked_lm_weights":
        tf.io.FixedLenFeature([76], tf.float32),
    "next_sentence_labels":
        tf.io.FixedLenFeature([1], tf.int64),
}

def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature


def create_float_feature(values):
  feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
  return feature


    # features = collections.OrderedDict()
    # features["input_ids"] = create_int_feature(input_ids)
    # features["input_mask"] = create_int_feature(input_mask)
    # features["segment_ids"] = create_int_feature(segment_ids)
    # features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
    # features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
    # features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
    # features["next_sentence_labels"] = create_int_feature([next_sentence_label])
    # # Make an example (case) out of the above feature dictionary
    # # https://www.tensorflow.org/tutorials/load_data/tfrecord#creating_a_tftrainexample_message
    # tf_example = tf.train.Example(features=tf.train.Features(feature=features))

def _decode_record(record):
  """Decodes a record to a TensorFlow example."""
  return tf.io.parse_single_example(record, NAME_TO_FEATURES)

  # # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # # So cast all int64 to int32.
  # for name in list(example.keys()):
  #   t = example[name]
  #   if t.dtype == tf.int64:
  #     t = tf.to_int32(t)
  #   example[name] = t

def to_tf_example(parsed_record):
  features = collections.OrderedDict()
  features["input_ids"] = create_int_feature(parsed_record['input_ids'])
  features["input_mask"] = create_int_feature(parsed_record['input_mask'])
  features["segment_ids"] = create_int_feature(parsed_record['segment_ids'])
  features["masked_lm_positions"] = create_int_feature(parsed_record['masked_lm_positions'])
  features["masked_lm_ids"] = create_int_feature(parsed_record['masked_lm_ids'])
  features["masked_lm_weights"] = create_float_feature(parsed_record['masked_lm_weights'])
  features["next_sentence_labels"] = create_int_feature([parsed_record['next_sentence_labels']])
  return tf.train.Example(features=tf.train.Features(feature=features))

def main(_):
  # tf.logging.set_verbosity(tf.logging.INFO)

  # tf.gfile.MakeDirs(FLAGS.output_dir)

  input_files = []

  # This returns the file names matching the given glob pattern
  # Aka. will fetch the names of all input tfrecords
  for input_pattern in FLAGS.input_file.split(","):
    input_files.extend(tf.io.gfile.glob(input_pattern))
  input_files=input_files[0:10]

  # tf.logging.info("*** Input Files ***")
  # for input_file in input_files:
    # tf.logging.info("  %s" % input_file)


  raw_dataset = tf.data.TFRecordDataset(input_files)

  for raw_record in raw_dataset.take(1):
    print(raw_record)
    print(tf.size(raw_record))

  parsed_dataset = raw_dataset.map(_decode_record).repeat()

  for parsed_record in parsed_dataset.take(1):
    import code 
    code.interact(local=locals())

  exit()
  # python shorten_dataset.py --input_file=/data/part* --output_dir=/output
  
  # Create shorter files 
  for i in range(10):
    # tf.logging.info(f"Creating shortened file {i}")
    outfile = os.path.join(FLAGS.output_dir, f"small_{i}.tfrecord")

    with tf.io.TFRecordWriter(outfile) as writer:
      for i, parsed_record in enumerate(parsed_dataset):
        print(i.shape)
        import code 
        code.interact(local=locals())
        if i == 10_000:
          break
        tf_example = to_tf_example(parsed_record)
        writer.write(tf_example.SerializeToString())
        exit()

  tf.logging.info("Done")

  input_files = []
  # Try opening the written files
  # This returns the file names matching the given glob pattern
  # Aka. will fetch the names of all input tfrecords
  tf.logging.info("Testing writen files")
  for input_pattern in os.path.join(FLAGS.output_dir, f"small_*"):
    input_files.extend(tf.gfile.Glob(input_pattern))

  tf.logging.info("*** Input Files ***")
  for input_file in input_files:
    tf.logging.info("  %s" % input_file)

  raw_dataset = tf.data.TFRecordDataset(input_files)

  for raw_record in raw_dataset.take(10):
    print(repr(raw_record))

  parsed_dataset = raw_dataset.map(_decode_record).repeat()

  for parsed_record in parsed_dataset.take(2):
    print(repr(parsed_record))


if __name__ == "__main__":
  flags.mark_flag_as_required("input_file")
  # flags.mark_flag_as_required("output_dir")
  absl.app.run(main)
