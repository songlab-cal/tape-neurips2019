from typing import Dict, Optional, Tuple
import string
import tensorflow as tf
from tqdm import tqdm

import pickle as pkl

from .tf_data_utils import to_features, to_sequence_features


def parse_line(line):
    pdb1, pdb2, seq1, seq2, scop1, scop2, label = line.split('\t')
    label = label.strip()

    return pdb1, pdb2, seq1, seq2, scop1, scop2, label


def convert_paired_sequences_to_tfrecords(train: str,
                                          val: str,
                                          test: str,
                                          outfile: Optional[str],
                                          vocab: Optional[Dict[str, int]] = None) -> None:
    if outfile is None:
        outfile = 'berger_SCOPe'
    else:
        outfile = outfile.rsplit('.')[0]

    if vocab is None:
        vocab = {"<PAD>": 0, "<MASK>": 1, "<CLS>": 2, "<SEP>": 3}

    with open(train) as data_file:
        train_sequences = data_file.readlines()

    with open(val) as data_file:
        val_sequences = data_file.readlines()

    with open(test) as data_file:
        test_sequences = data_file.readlines()

    i = 0
    with tf.python_io.TFRecordWriter(outfile + '_train.tfrecords') as writer:
        print('Creating Training Set...')
        for example in tqdm(train_sequences):
            try:
                features = parse_line(example)
                serialized_example, vocab = serialize_paired_sequence(features, vocab)
                writer.write(serialized_example)
            except ValueError:
                i += 1
                pass

    with tf.python_io.TFRecordWriter(outfile + '_valid.tfrecords') as writer:
        print('Creating Validation Set...')
        for example in tqdm(val_sequences):
            try:
                features = parse_line(example)

                serialized_example, vocab = serialize_paired_sequence(features, vocab)
                writer.write(serialized_example)
            except ValueError:
                i += 1
                pass

    with tf.python_io.TFRecordWriter(outfile + '_test.tfrecords') as writer:
        print('Creating Testing Set...')
        for example in tqdm(test_sequences):
            try:
                features = parse_line(example)

                serialized_example, vocab = serialize_paired_sequence(features, vocab)
                writer.write(serialized_example)
            except ValueError:
                i += 1
                pass

    with open(outfile.rsplit('.', maxsplit=1)[0] + '.vocab', 'wb') as f:
        pkl.dump(vocab, f)


def serialize_paired_sequence(features: Tuple[str, ...], vocab: Dict[str, int]) -> Tuple[bytes, Dict[str, int]]:
    pdb1, pdb2, seq1, seq2, scop1, scop2, label = features

    int_seq1 = []
    for aa in seq1:
        if aa in string.whitespace:
            raise ValueError("whitespace found in string")

        aa_idx = vocab.get(aa.upper())
        if aa_idx is None:
            raise ValueError(f"{aa} is not in vocab")

        int_seq1.append(aa_idx)

    int_seq2 = []
    for aa in seq2:
        if aa in string.whitespace:
            raise ValueError("whitespace found in string")

        aa_idx = vocab.get(aa.upper())
        if aa_idx is None:
            raise ValueError(f"{aa} is not in vocab")

        int_seq2.append(aa_idx)

    protein_context = to_features(
        # Metadata for convenience
        first_scop=scop1.encode('UTF-8'),
        second_scop=scop2.encode('UTF-8'),
        first_pdb=pdb1.encode('UTF-8'),
        second_pdb=pdb2.encode('UTF-8'),
        # Context for training
        first_length=len(int_seq1),
        second_length=len(int_seq2),
        label=int(label))

    # Sequences
    protein_features = to_sequence_features(
        first_sequence=int_seq1,
        second_sequence=int_seq2)

    example = tf.train.SequenceExample(context=protein_context, feature_lists=protein_features)
    return example.SerializeToString(), vocab


def deserialize_paired_sequence(example):
    context = {
        'first_scop': tf.FixedLenFeature([], tf.string),
        'second_scop': tf.FixedLenFeature([], tf.string),
        'first_pdb': tf.FixedLenFeature([], tf.string),
        'second_pdb': tf.FixedLenFeature([], tf.string),
        'first_length': tf.FixedLenFeature([1], tf.int64),
        'second_length': tf.FixedLenFeature([1], tf.int64),
        'label': tf.FixedLenFeature([1], tf.int64)
    }

    features = {
        'first_sequence': tf.FixedLenSequenceFeature([1], tf.int64),
        'second_sequence': tf.FixedLenSequenceFeature([1], tf.int64)
    }

    context, features = tf.parse_single_sequence_example(
        example,
        context_features=context,
        sequence_features=features
    )

    first_scop = context['first_scop']
    second_scop = context['second_scop']
    first_pdb = context['first_pdb']
    second_pdb = context['second_pdb']

    first_length = tf.to_int32(context['first_length'][0])
    second_length = tf.to_int32(context['second_length'][0])
    label = tf.cast(context['label'], tf.int32)
    first_sequence = tf.to_int32(features['first_sequence'][:, 0])
    second_sequence = tf.to_int32(features['second_sequence'][:, 0])

    first = {
        'scop': first_scop,
        'pdb': first_pdb,
        'protein_length': first_length,
        'sequence': first_sequence}

    second = {
        'scop': second_scop,
        'pdb': second_pdb,
        'protein_length': second_length,
        'sequence': second_sequence}

    return {'first': first, 'second': second, 'label': label}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='convert protein sequences to tfrecords')
    parser.add_argument('filename', type=str, help='text file to convert to tfrecords')
    parser.add_argument('--outfile', type=str, default=None, help='name of outfile')
    parser.add_argument('--vocab', type=str, default=None, help='path to existing vocab file')
    args = parser.parse_args()

    endings = ['train.sampledpairs.txt', 'train.dev.sampledpairs.txt', 'test.sampledpairs.txt']
    filenames = [args.filename + '.' + end for end in endings]
    train, val, test = filenames

    if args.vocab:
        with open(args.vocab, 'rb') as f:
            vocab = pkl.load(f)

    convert_paired_sequences_to_tfrecords(train, val, test, args.outfile, vocab)
