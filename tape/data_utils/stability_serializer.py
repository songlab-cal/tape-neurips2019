from typing import Dict
from functools import partial
from multiprocessing import Pool
import os
import string

import pandas as pd
import numpy as np
import tensorflow as tf

from .vocabs import PFAM_VOCAB
from .tf_data_utils import to_features, to_sequence_features

"""
Inputs
    primary: The amino acid sequence of the protein.
    protein_length: The length of the actual sequence, used for padding and masking.

Outputs
    stability_score: Scalar used for regression.

Metadata
    parent: If a protein is a variant, indicates which protein it is a variant of. Only applicable for the test-set.
    topology: Topology of the protein. There are seven possibilities.
    id: The ID of the protein, drawn from the Rocklin et al paper. Of the form topology_round#_uniqueid.pdb
`

"""
filenames = ['rd1_stability_scores',
             'rd2_stability_scores',
             'rd3_stability_scores',
             'rd4_stability_scores',
             'ssm2_stability_scores']


designed_topologies = ['EEHEE', 'EHEE', 'HEEH', 'HHH']
all_topologies = ['EEHEE', 'EHEE', 'HEEH', 'HHH', 'Pin1', 'hYAP65', 'villin']

# 12197 rd1_stability_scores
# 11770 rd2_stability_scores
# 12522 rd3_stability_scores
# 19698 rd4_stability_scores
# 12852 ssm2_stability_scores
# 69039 total


def make_train_val_test_split(rd1, rd2, rd3, rd4, ssm2):
    # any rd4 sequence derived from any other previous sequence is out
    # EEHEE_rd1_0001.pdb
    # EEHEE_rd1_0001.pdb_hp
    # EEHEE_rd1_0001.pdb_random
    # other modifiers include
    # '_PG_hp'
    # '_buryD'
    # '_PG_hp_prottest_XXX'

    base_name = rd1['name'].str.split('.', n=1, expand=True)
    rd1['base_name'] = base_name[0]
    topology = rd1['base_name'].str.split('_', n=1, expand=True)
    rd1['topology'] = topology[0]

    base_name = rd2['name'].str.split('.', n=1, expand=True)
    rd2['base_name'] = base_name[0]
    topology = rd2['base_name'].str.split('_', n=1, expand=True)
    rd2['topology'] = topology[0]

    base_name = rd3['name'].str.split('.', n=1, expand=True)
    rd3['base_name'] = base_name[0]
    topology = rd3['base_name'].str.split('_', n=1, expand=True)
    rd3['topology'] = topology[0]

    base_name = rd4['name'].str.split('.', n=1, expand=True)
    rd4['base_name'] = base_name[0]
    topology = rd4['base_name'].str.split('_', n=1, expand=True)
    rd4['topology'] = topology[0]

    base_name = ssm2['name'].str.split('.', n=1, expand=True)
    ssm2['base_name'] = base_name[0]
    topology = ssm2['base_name'].str.split('_', n=1, expand=True)
    ssm2['topology'] = topology[0]

    # need to filter out all sequences from val based on the original ones...
    all_base = list(rd1.base_name.values)
    all_base.extend(rd2.base_name.values)
    all_base.extend(rd3.base_name.values)

    train = rd1
    train = train.append(rd2)
    train = train.append(rd3)

    # filter 1552 sequences that appear in training already
    train = train.append(rd4[rd4['base_name'].isin(all_base)])
    # 18145 remaining
    val_set = rd4[~rd4['base_name'].isin(all_base)]

    validation = pd.DataFrame()
    for topology in designed_topologies:
        top_set = val_set[val_set['topology'] == topology]
        # pick 200 base sequences for val
        base_seqs = np.random.choice(top_set.base_name.values, size=200)
        # use the base sequences + controls (buryD, PG_hp) for validation ~500
        val_for_topology = top_set[top_set['base_name'].isin(base_seqs)]
        validation = validation.append(val_for_topology)
        print('validation for topology {}'.format(topology))
        print(val_for_topology.shape[0])
        to_train = top_set[~top_set['base_name'].isin(base_seqs)]
        print(to_train.shape[0])
        train = train.append(to_train)

    # 5k more to train on that are not part of the designed topologies
    train = train.append(val_set[~val_set['topology'].isin(designed_topologies)])

    test = ssm2
    return train, validation, test


def serialize_stability_sequence(sequence: str,
                                 seq_id: str,
                                 topology: str,
                                 parent: str,
                                 stabilityscore_c: float,
                                 vocab: Dict[str, int]):
    int_sequence = []
    for aa in sequence:
        if aa in string.whitespace:
            raise ValueError("whitespace found in string")

        aa_idx = vocab.get(aa)
        if aa_idx is None:
            raise ValueError(f'{aa} not in vocab')

        int_sequence.append(aa_idx)

    protein_context = {}
    protein_context = to_features(
        id=seq_id.encode('UTF-8'),
        protein_length=len(int_sequence),
        topology=topology.encode('UTF-8'),
        parent=parent.encode('UTF-8'),
        stability_score=stabilityscore_c)

    protein_features = to_sequence_features(primary=int_sequence)

    example = tf.train.SequenceExample(context=protein_context, feature_lists=protein_features)
    return example.SerializeToString()


def deserialize_stability_sequence(example):
    context = {
        'id': tf.FixedLenFeature([], tf.string),
        'protein_length': tf.FixedLenFeature([1], tf.int64),
        'topology': tf.FixedLenFeature([], tf.string),
        'parent': tf.FixedLenFeature([], tf.string),
        'stability_score': tf.FixedLenFeature([1], tf.float32)
    }

    features = {
        'primary': tf.FixedLenSequenceFeature([1], tf.int64),
    }

    context, features = tf.parse_single_sequence_example(
        example,
        context_features=context,
        sequence_features=features
    )

    sequence = tf.to_int32(features['primary'][:, 0])
    protein_length = tf.to_int32(context['protein_length'][0])
    stability_score = tf.cast(context['stability_score'], tf.float32)

    return {'id': context['id'],
            'primary': sequence,
            'protein_length': protein_length,
            'topology': context['topology'],
            'parent': context['parent'],
            'stability_score': stability_score}


def convert_round_outputs_to_tfrecords(datadir, outdir, vocab):
    rd1 = pd.read_csv(os.path.join(datadir, filenames[0]), sep='\s+')
    rd2 = pd.read_csv(os.path.join(datadir, filenames[1]), sep='\s+')
    rd3 = pd.read_csv(os.path.join(datadir, filenames[2]), sep='\s+')
    rd4 = pd.read_csv(os.path.join(datadir, filenames[3]), sep='\s+')
    ssm2 = pd.read_csv(os.path.join(datadir, filenames[4]), sep='\s+')
    train, valid, test = make_train_val_test_split(rd1, rd2, rd3, rd4, ssm2)

    print('train {} valid {} test {}'.format(train.shape[0], valid.shape[0], test.shape[0]))

    train = train.dropna(subset=['stabilityscore_c', 'stabilityscore_t'])
    valid = valid.dropna(subset=['stabilityscore_c', 'stabilityscore_t'])
    test = test.dropna(subset=['stabilityscore_c', 'stabilityscore_t'])

    print('after dropping NAs')
    print('train {} valid {} test {}'.format(train.shape[0], valid.shape[0], test.shape[0]))

    train_filename = os.path.join(outdir, 'stability_train.tfrecords')
    valid_filename = os.path.join(outdir, 'stability_valid.tfrecords')
    test_filename = os.path.join(outdir, 'stability_test.tfrecords')

    make_tfrecord(train, train_filename, vocab)
    make_tfrecord(valid, valid_filename, vocab)
    make_tfrecord(test, test_filename, vocab)


def make_tfrecord(df, filename, vocab):
    serialize_with_vocab = partial(serialize_stability_sequence, vocab=vocab)

    to_tfrecord = []
    for index, row in df.iterrows():
        to_tfrecord.append([row['sequence'], row['name'], row['topology'], row['base_name'], row['stabilityscore_c']])

    print('Serializing {} examples...'.format(len(to_tfrecord)))
    with Pool() as p:
        serialized_examples = p.starmap(serialize_with_vocab, to_tfrecord)

    with tf.python_io.TFRecordWriter(filename) as writer:
        print('Creating {}...'.format(filename))
        for serialized_example in serialized_examples:
            writer.write(serialized_example)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='convert protein engineering results to tfrecords')
    parser.add_argument('--datadir', required=True, help='where the *_stability_scores files are')
    parser.add_argument('--outdir', type=str, help='name of outdir')
    args = parser.parse_args()

    vocab = PFAM_VOCAB

    convert_round_outputs_to_tfrecords(args.datadir, args.outdir, vocab)
