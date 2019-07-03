import string

import tensorflow as tf

from Bio import SeqIO

from tape.data_utils.tf_data_utils import to_features, to_sequence_features
from tape.data_utils.vocabs import PFAM_VOCAB

filename = 'small.fasta'
vocab = PFAM_VOCAB


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

        protein_context = to_features(protein_length=len(int_sequence), description=description.encode('UTF-8'))
        protein_features = to_sequence_features(primary=int_sequence)

        example = tf.train.SequenceExample(context=protein_context, feature_lists=protein_features)
        proteins_to_embed.append(example)

    new_filename = filename.rsplit('.', maxsplit=1)[0] + '.tfrecord'

    print('Writing tfrecord file {}'.format(new_filename))

    with tf.python_io.TFRecordWriter(new_filename) as writer:
        writer.write(example.SerializeToString())


if __name__ == '__main__':
    fasta_to_tfrecord
    import argparse
    parser = argparse.ArgumentParser(description='convert protein sequences to tfrecords')
    parser.add_argument('filename', type=str, help='text file to convert to tfrecords')
    args = parser.parse_args()

    fasta_to_tfrecord(args.filename)
