import string

import tensorflow as tf

from Bio import SeqIO

from tape.data_utils.tf_data_utils import to_features, to_sequence_features
from tape.data_utils.vocabs import PFAM_VOCAB


def fasta_to_tfrecord(filename, vocab=PFAM_VOCAB):
    proteins_to_embed = []
    for record in SeqIO.parse(filename, 'fasta'):
        sequence = record.seq
        description = record.description

        int_sequence = []
        for aa in sequence:
            if aa in string.whitespace:
                raise ValueError("whitespace found in sequence for {}".format(description))
            int_sequence.append(vocab.get(aa))

        protein_context = to_features(protein_length=len(int_sequence), id=description.encode('UTF-8'))
        protein_features = to_sequence_features(primary=int_sequence)

        example = tf.train.SequenceExample(context=protein_context, feature_lists=protein_features)
        proteins_to_embed.append(example)

    new_filename = filename.rsplit('.', maxsplit=1)[0] + '.tfrecord'

    print('Writing tfrecord file {}'.format(new_filename))

    with tf.python_io.TFRecordWriter(new_filename) as writer:
        writer.write(example.SerializeToString())


def deserialize_protein_sequence(example):
    context = {
        'protein_length': tf.FixedLenFeature([1], tf.int64),
        'id': tf.FixedLenFeature([], tf.string)
    }

    features = {
        'primary': tf.FixedLenSequenceFeature([1], tf.int64),
    }

    context, features = tf.parse_single_sequence_example(
        example,
        context_features=context,
        sequence_features=features
    )

    return {'id': context['id'],
            'primary': tf.to_int32(features['primary'][:, 0]),
            'protein_length': tf.to_int32(context['protein_length'][0])}


def main():
    import argparse
    parser = argparse.ArgumentParser(description='convert protein sequences to tfrecords')
    parser.add_argument('filename', type=str, help='text file to convert to tfrecords')
    args = parser.parse_args()

    fasta_to_tfrecord(args.filename)


if __name__ == '__main__':
    main()
