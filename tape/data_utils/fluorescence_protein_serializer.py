from typing import Dict, Optional, Tuple
import random
import string
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle as pkl

from .vocabs import PFAM_VOCAB

"""
Inputs
    primary: The amino acid sequence of the protein.
    protein_length: The length of the actual sequence, used for padding and masking.

Outputs
    log_fluorescence: The log-fluorescence of the protein, used for regression.

Metadata
    num_mutations: The number of mutations (Hamming distance) of the protein from initial green fluorescent protein.
"""

def _bytes_feature(value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=value)
    )


def _int64_feature(value):
    return tf.train.Feature(
        int64_list=tf.train.Int64List(value=value)
    )


def _float_feature(value):
    return tf.train.Feature(
        float_list=tf.train.FloatList(value=value)
    )


def to_sequence_features(**features):
    for name, array in features.items():
        if array.dtype in [np.int32, np.int64]:
            array = np.asarray(array, np.int32)
            array = [_int64_feature(el) for el in array]
        elif array.dtype in [np.float32, np.float64]:
            array = np.asarray(array, np.float32)
            array = [_float_feature(el) for el in array]
        else:
            raise TypeError("Unrecognized dtype {}. Can only handle int or float dtypes.".format(array.dtype))
        features[name] = tf.train.FeatureList(feature=array)

    features = tf.train.FeatureLists(feature_list=features)
    return features


def convert_fluorescence_sequences_to_tfrecords(filename: str,
                                       threshold: int,
                                       outfile: Optional[str],
                                       vocab: Optional[Dict[str, int]] = None) -> None:
    if outfile is None:
        outfile = filename.rsplit('.')[0]
    else:
        outfile = outfile.rsplit('.')[0]

    if vocab is None:
        vocab = {"<PAD>": 0, "<MASK>": 1, "<CLS>": 2, "<SEP>": 3}

    df = pd.read_csv(filename)

    # Use threshold to create test set
    train = df[df.num_mutations <= threshold]
    test = df[df.num_mutations > threshold].reset_index(drop=True)

    # Now make a train/val split on shuffled train set
    train = train.sample(frac=1)
    train_split = int(train.shape[0] * 0.8)
    valid = train.tail(train.shape[0] - train_split).reset_index(drop=True)
    train = train.head(train_split).reset_index(drop=True)

    with tf.python_io.TFRecordWriter(outfile + '_train.tfrecords') as writer:
        print('Creating Training Set...')
        for example in tqdm(train.iterrows()):
            __, example = example
            serialized_example = serialize_fluorescence_sequence(tuple(example), vocab)
            writer.write(serialized_example)

    with tf.python_io.TFRecordWriter(outfile + '_valid.tfrecords') as writer:
        print('Creating Validation Set...')
        for example in tqdm(valid.iterrows()):
            __, example = example
            serialized_example = serialize_fluorescence_sequence(tuple(example), vocab)
            writer.write(serialized_example)

    with tf.python_io.TFRecordWriter(outfile + '_test.tfrecords') as writer:
        print('Creating Test Set...')
        for example in tqdm(test.iterrows()):
            __, example = example
            serialized_example = serialize_fluorescence_sequence(tuple(example), vocab)
            writer.write(serialized_example)


def serialize_fluorescence_sequence(features: Tuple[str, float, int], vocab: Dict[str, int]) -> bytes:
    sequence, num_mutations, y = features

    int_sequence = []
    for aa in sequence:
        if aa == '*':
            continue
        elif aa in string.whitespace:
            raise ValueError("whitespace found in string")

        aa_idx = vocab.get(aa)
        if aa_idx is None:
            raise ValueError(f"{aa} not in vocab")

        int_sequence.append(aa_idx)

    protein_context = {}
    protein_context['protein_length'] = _int64_feature([len(int_sequence)])
    protein_context['log_fluorescence'] = _float_feature([y])
    protein_context['num_mutations'] = _int64_feature([num_mutations])
    protein_context = tf.train.Features(feature=protein_context)

    protein_features = {}
    protein_features['primary'] = [_int64_feature([el]) for el in int_sequence]

    for key, val in protein_features.items():
        protein_features[key] = tf.train.FeatureList(feature=val)
    protein_features = tf.train.FeatureLists(feature_list=protein_features)

    example = tf.train.SequenceExample(context=protein_context, feature_lists=protein_features)
    return example.SerializeToString()


def deserialize_fluorescence_sequence(example):
    context = {
        'protein_length': tf.FixedLenFeature([1], tf.int64),
        'log_fluorescence': tf.FixedLenFeature([1], tf.float32),
        'num_mutations': tf.FixedLenFeature([1], tf.int64)
    }

    features = {
        'primary': tf.FixedLenSequenceFeature([1], tf.int64),
    }

    context, features = tf.parse_single_sequence_example(
        example,
        context_features=context,
        sequence_features=features
    )

    protein_length = tf.cast(context['protein_length'][0], tf.int32)
    log_fluorescence = context['log_fluorescence']
    num_mutations = tf.cast(context['num_mutations'][0], tf.int32)
    sequence = tf.to_int32(features['primary'][:, 0])

    return {'primary': sequence,
            'protein_length': protein_length,
            'log_fluorescence': log_fluorescence,
            'num_mutations': num_mutations}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='convert protein sequences to tfrecords')
    parser.add_argument('filename', type=str, help='text file to convert to tfrecords')
    parser.add_argument('--threshold', type=int, default = 3, help='number of mutation threshold for separation')
    parser.add_argument('--outfile', type=str, default=None, help='name of outfile')
    args = parser.parse_args()

    vocab = PFAM_VOCAB
    convert_fluorescence_sequences_to_tfrecords(args.filename, args.threshold, args.outfile, vocab)
