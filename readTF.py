import argparse
import random
import tensorflow as tf
import numpy as np
import collections

max_seq_length=512
max_predictions_per_seq=76

def read_TFrecord(input):


    max_seq_length=512
    max_predictions_per_seq=76

    feature_description = {
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

    tf.enable_eager_execution()

    raw_dataset = tf.data.TFRecordDataset(input)
    print(sum(1 for _ in tf.python_io.tf_record_iterator(input)))

    #for raw_record in raw_dataset.take(10):
    print(repr(raw_dataset))

    def _parse_function(example_proto):
    # Parse the input `tf.train.Example` proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, feature_description)
    
    parsed_dataset = raw_dataset.map(_parse_function)
    for parsed_record in parsed_dataset.take(2):
        print(repr(parsed_record))


def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature

def create_float_feature(values):
  feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
  return feature



# input ids: 30522 (from vocab.txt)


def write_instance_to_example_files():

    # also select length?


    id_length = random.randint(0, max_seq_length)
    input_ids = [random.randint(0, 30523) for _ in range(id_length)] + [0 for _ in range(max_seq_length - id_length)]
    input_mask = [1 for _ in range(id_length)] + [0 for _ in range(max_seq_length - id_length)]

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(input_ids)
    features["input_mask"] = create_int_feature(input_mask)
    features["segment_ids"] = create_int_feature(segment_ids)
    features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
    features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
    features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
    features["next_sentence_labels"] = create_int_feature([next_sentence_label])





# def serialize_example(input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights, next_sentence_labels):
#   """
#   Creates a tf.train.Example message ready to be written to a file.
#   """
#   # Create a dictionary mapping the feature name to the tf.train.Example-compatible
#   # data type.
#   feature = {
#     "input_ids": _int64_feature(input_ids),
#     "input_mask": _int64_feature(input_mask),
#     "segment_ids": _int64_feature(segment_ids),
#     "masked_lm_positions": _float_feature(masked_lm_positions),
#     "masked_lm_ids": _int64_feature(masked_lm_ids),
#     "masked_lm_weights": _float_feature(masked_lm_weights),
#     "next_sentence_labels": _int64_feature(next_sentence_labels)

#   }

#   # Create a Features message using tf.train.Example.

#   example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
#   return example_proto.SerializeToString()


def generate_data_bert():

    # could randomly pick 0's or 1's

    # 224732 number of instances
    tf.enable_eager_execution()

    

    n_observations = 10

    input_ids = np.tile(np.arange(1, max_seq_length+1, dtype=np.int64),(n_observations,1))
    input_mask = np.tile(np.ones((max_seq_length,), dtype=np.int64), (n_observations, 1))
    segment_ids = np.tile(np.zeros((max_seq_length,), dtype=np.int64),(n_observations, 1))
    masked_lm_positions = np.tile(np.arange(1, max_predictions_per_seq+1, dtype=np.int64), (n_observations, 1))
    masked_lm_ids = np.tile(np.arange(1, max_predictions_per_seq+1, dtype=np.int64),(n_observations, 1))
    masked_lm_weights = np.tile(np.ones((max_predictions_per_seq,), dtype=np.float32), (n_observations, 1))
    next_sentence_labels = np.tile(np.array([1]), (n_observations, 1))

    features_dataset = tf.data.Dataset.from_tensor_slices((input_ids, input_mask, segment_ids, masked_lm_positions,  masked_lm_ids,  masked_lm_weights, next_sentence_labels))
    for f in features_dataset.take(1):
        print(f)




if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--input', dest='input', required=True)
    args = PARSER.parse_args()
    read_TFrecord(args.input)
    # generate_data_bert()

  