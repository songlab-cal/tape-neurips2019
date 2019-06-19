from typing import Dict, Optional, Tuple
import random
import string
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import re
from os import listdir
from os.path import isfile, join
import pickle as pkl
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO


""" 
Paired Homology Detection task introduced by Bepler et al in https://openreview.net/pdf?id=SygLehCqtm
Used for supervised pre-training of the Bepler model. Not currently guaranteed to run.
"""

def form_binary_task_dict(folder: str, is_fold:bool):
    """Extracts names of all tasks in a folder and saves in simple dict.
    """
    files = [f for f in listdir(folder) if isfile(join(folder, f))]
    # Hack to extract folds vs superfamilies using SCOP notation
    if is_fold:
        n_split = 3
    else:
        n_split = 4
    extract_name = lambda x: '.'.join(x.split('.')[1:n_split+1])
    task_names = [extract_name(file) for file in files]
    task_names = list(set(task_names))
    task_dict = {i:name for i,name in enumerate(task_names)}

    if is_fold:
        dict_name = 'scop_binary_fold_names.pkl'
    else:
        dict_name = 'scop_binary_superfamily_names.pkl'
    with open(dict_name, 'wb') as f:
        pkl.dump(task_dict, f)

    return files, task_names

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

def parse_binary_scop_line(record: SeqRecord) -> Tuple[str, str]:
    seq = record.seq
    seq_id = record.id

    return str(seq), seq_id

def convert_binary_scop_task_to_tfrecords(filename: str,
                                   task_name: str,
                                   outfile: Optional[str],
                                   vocab: Optional[Dict[str, int]] = None) -> None:

    # Create the list of train/test split for given task
    prefixes = ['pos-train', 'neg-train', 'pos-test', 'neg-test']
    prefixes = [filename + '/' + prefix for prefix in prefixes]
    files = [prefix + '.' + task_name + '.fasta' for prefix in prefixes]

    if vocab is None:
        vocab = {"<PAD>": 0, "<MASK>": 1, "<CLS>": 2, "<SEP>": 3}

    # Load train / valid positive examples and negative examples separately
    datasets = []
    for i, file in enumerate(files):
        datasets.append([])
        for record in SeqIO.parse(file, 'fasta'):
            # Store each example as a [seq, seq_id, label] triple
            seq, seq_id = parse_binary_scop_line(record)
            label = 1 if i%2 == 0 else 0
            datasets[i].append([seq, seq_id, label])

    # Merge pos and negative examples for train and valid, then shuffle
    train = datasets[0] + datasets[1]
    valid = datasets[2] + datasets[3]
    random.shuffle(train)
    random.shuffle(valid)

    with tf.python_io.TFRecordWriter(outfile + '_train.tfrecords') as writer:
        print('Creating Training Set...')
        for example in tqdm(train):
            seq, seq_id, label = example

            serialized_example = serialize_binary_scop_sequence(seq.strip(), seq_id, label, vocab)
            writer.write(serialized_example)

    with tf.python_io.TFRecordWriter(outfile + '_valid.tfrecords') as writer:
        print('Creating Validation Set...')
        for example in tqdm(valid):
            seq, seq_id, label = example

            serialized_example = serialize_binary_scop_sequence(seq.strip(), seq_id, label, vocab)
            writer.write(serialized_example)

def serialize_binary_scop_sequence(sequence: str, seq_id: str, label: int, vocab: Dict[str, int]) -> bytes:
    int_sequence = []
    for aa in sequence:
        if aa in string.whitespace:
            raise ValueError("whitespace found in string")

        aa_idx = vocab.get(aa)
        if aa_idx is None:
            raise ValueError(f'{aa} not in vocab')

        int_sequence.append(aa_idx)

    protein_context = {}
    protein_context['sequence_id'] = _bytes_feature([seq_id.encode('UTF-8')])
    protein_context['protein_length'] = _int64_feature([len(int_sequence)])
    protein_context['label'] = _int64_feature([label])
    protein_context = tf.train.Features(feature=protein_context)

    protein_features = {}
    protein_features['sequence'] = [_int64_feature([el]) for el in int_sequence]

    for key, val in protein_features.items():
        protein_features[key] = tf.train.FeatureList(feature=val)
    protein_features = tf.train.FeatureLists(feature_list=protein_features)

    example = tf.train.SequenceExample(context=protein_context, feature_lists=protein_features)
    return example.SerializeToString()


def deserialize_binary_scop_sequence(example):
    context = {
        'sequence_id': tf.FixedLenFeature([], tf.string),
        'protein_length': tf.FixedLenFeature([1], tf.int64),
        'label': tf.FixedLenFeature([1], tf.int64)
    }

    features = {
        'sequence': tf.FixedLenSequenceFeature([1], tf.int64),
    }

    context, features = tf.parse_single_sequence_example(
        example,
        context_features=context,
        sequence_features=features
    )

    sequence_id = context['sequence_id']
    protein_length = tf.to_int32(context['protein_length'][0])
    label = tf.cast(context['label'], tf.int32)
    sequence = tf.to_int32(features['sequence'][:, 0])

    return {'sequence_id': sequence_id,
            'sequence': sequence,
            'protein_length': protein_length,
            'label': label}



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='convert protein sequences to tfrecords')
    parser.add_argument('filename', type=str, help='text file to convert to tfrecords')
    parser.add_argument('task_dict', type=str, help='path to dict of task labels')
    parser.add_argument('--outfile', type=str, default=None, help='name of outfile')
    parser.add_argument('--vocab', type=str, help='path to vocab file')
    args = parser.parse_args()

    # Get dict of all tasks
    with open(args.task_dict, 'rb') as f:
        task_dict = pkl.load(f)

    with open(args.vocab, 'rb') as f:
        vocab = pkl.load(f)

    for k, task_name in task_dict.items():
        print(f'Task {k} is {task_name}')
        task_outfile = args.outfile + '_' + str(k)
        convert_binary_scop_task_to_tfrecords(args.filename, task_name, task_outfile, vocab)
