import os
from re import I
import absl
import numpy as np

import collections
import tensorflow as tf

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
  tf.logging.set_verbosity(tf.logging.INFO)

  input_files = []

  # This returns the file names matching the given glob pattern
  # Aka. will fetch the names of all input tfrecords
  input_files.extend(tf.gfile.Glob('/data/part-*')).sort()

  print(input_files)

  records_per_file = []

  with open('/output/records_per_file.txt', 'w') as outfile:
    for file in input_files:
      dataset = tf.data.TFRecordDataset(file)
      dataset = dataset.map(_decode_record)

      num_records = 0
      print(f'File {file}:')
      outfile.write(f'File {file}\n')

      for _ in dataset:
        num_records += 1

      print(f'Num records: {num_records}')
      outfile.write(f'Num records: {num_records}\n')
      outfile.flush()

    records_per_file = np.asarray(records_per_file)
    print(f'Average num samples per file: {records_per_file.mean()} - std {records_per_file.std()}')
    outfile.write(f'Average num samples per file: {records_per_file.mean()} - std {records_per_file.std()}\n')



if __name__ == "__main__":
  absl.app.run(main)
