import argparse
from contextlib import suppress

import tensorflow as tf
import numpy as np

from Bio import SeqIO

from tape.data_utils.vocabs import PFAM_VOCAB
from tape.data_utils.serialize_fasta import deserialize_fasta_sequence
from tape.models import ModelBuilder


def embed_from_fasta(fasta_file, model: str, load_from=None, vocab=PFAM_VOCAB):
    sess = tf.Session()
    embedding_model = ModelBuilder.build_model(model)

    sequence = tf.placeholder(tf.int32, [None, None])
    protein_length = tf.placeholder(tf.int32, [None])
    output = embedding_model({'sequence': sequence, 'protein_length': protein_length})

    if load_from is not None:
        embedding_model.load_weights(load_from)

    embeddings = []
    for record in SeqIO.parse(fasta_file, 'fasta'):
        int_sequence = np.array([vocab[aa] for aa in record.seq])
        encoder_output = sess.run(output['encoder_output'], feed_dict={sequence: int_sequence[None], protein_length: int_sequence.shape[1:]})
        embeddings.append(encoder_output)
    return embeddings


def embed_from_tfrecord(tfrecord_file, model: str, load_from=None, vocab=PFAM_VOCAB):
    sess = tf.Session()
    embedding_model = ModelBuilder.build_model(model)

    sequence = tf.placeholder(tf.int32, [None, None])
    protein_length = tf.placeholder(tf.int32, [None])
    output = embedding_model({'sequence': sequence, 'protein_length': protein_length})

    if load_from is not None:
        embedding_model.load_weights(load_from)

    data = tf.data.TFRecordDataset(tfrecord_file).map(deserialize_fasta_sequence)
    iterator = data.make_one_shot_iterator()
    batch = iterator.get_next()
    output = embedding_model(batch)
    embeddings = []
    with suppress(tf.errors.OutOfRangeError):
        while True:
            encoder_output = sess.run(output['encoder_output'])
            embeddings.append(encoder_output)
    return embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('datafile')
    parser.add_argument('--model', default='')
    parser.add_argument('--load-from', default=None)
    args = parser.parse_args()

    if args.datafile.endswith('.fasta'):
        embed_from_fasta(args.datafile, args.model, args.load_from)
    elif args.datafile.endswith('.tfrecord'):
        embed_from_tfrecord(args.datafile, args.model, args.load_from)
    else:
        raise Exception('Unsupported file type - only .fasta and .tfrecord supported')


if __name__ == '__main__':
    main()
