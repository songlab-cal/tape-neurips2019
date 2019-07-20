import argparse
from contextlib import suppress
import pickle as pkl

import tensorflow as tf
import numpy as np

from Bio import SeqIO

from tape.data_utils.vocabs import PFAM_VOCAB
from tape.data_utils.serialize_fasta import deserialize_fasta_sequence
from tape.models import ModelBuilder
from tape.tasks import TaskBuilder


def embed_from_fasta(fasta_file, model: str, load_from=None, vocab=PFAM_VOCAB):
    sess = tf.Session()
    embedding_model = ModelBuilder.build_model(model)

    primary = tf.placeholder(tf.int32, [None, None])
    protein_length = tf.placeholder(tf.int32, [None])
    output = embedding_model({'primary': primary, 'protein_length': protein_length})

    sess.run(tf.global_variables_initializer())
    if load_from is not None:
        embedding_model.load_weights(load_from)

    embeddings = []
    for record in SeqIO.parse(fasta_file, 'fasta'):
        int_sequence = np.array([vocab[aa] for aa in record.seq], ndmin=2)
        encoder_output = sess.run(output['encoder_output'],
                                  feed_dict={primary: int_sequence,
                                             protein_length: [int_sequence.shape[1]]})
        embeddings.append(encoder_output)
    return embeddings


def embed_from_tfrecord(tfrecord_file,
                        model: str,
                        load_from=None,
                        deserialization_func=deserialize_fasta_sequence):
    sess = tf.Session()
    embedding_model = ModelBuilder.build_model(model)

    primary = tf.placeholder(tf.int32, [None, None])
    protein_length = tf.placeholder(tf.int32, [None])
    output = embedding_model({'primary': primary, 'protein_length': protein_length})

    sess.run(tf.global_variables_initializer())
    if load_from is not None:
        embedding_model.load_weights(load_from)

    data = tf.data.TFRecordDataset(tfrecord_file).map(deserialization_func)
    data = data.batch(1)
    iterator = data.make_one_shot_iterator()
    batch = iterator.get_next()
    output = embedding_model(batch)
    embeddings = []
    with suppress(tf.errors.OutOfRangeError):
        while True:
            encoder_output_batch = sess.run(output['encoder_output'])
            for encoder_output in encoder_output_batch:
                embeddings.append(encoder_output)
    return embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('datafile')
    parser.add_argument('--model', default=None)
    parser.add_argument('--load-from', default=None)
    parser.add_argument('--task', default=None, help='If running a forward pass through existing task datasets, refer to the task with this flag')
    args = parser.parse_args()

    if args.task is not None:
        task = TaskBuilder.build_task(args.task)
        deserialization_func = task._deserialization_func
    else:
        deserialization_func = deserialize_fasta_sequence


    if args.datafile.endswith('.fasta'):
        embeddings = embed_from_fasta(args.datafile, args.model, args.load_from)
    elif args.datafile.endswith('.tfrecord'):
        embeddings = embed_from_tfrecord(args.datafile,
                                         args.model,
                                         args.load_from,
                                         deserialization_func)
    else:
        raise Exception('Unsupported file type - only .fasta and .tfrecord supported')

    with open('outputs.pkl', 'wb') as f:
        pkl.dump(embeddings, f)


if __name__ == '__main__':
    main()
