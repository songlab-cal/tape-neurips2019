from typing import Dict
import string
from functools import partial
from multiprocessing import Pool

import tensorflow as tf

from .tf_data_utils import to_features, to_sequence_features
from .bepler_contact_map_utils import get_contact_map_seqs, get_contact_map_paths
from .vocabs import PFAM_VOCAB

"""
Used for supervised pretraining as in Bepler et al https://openreview.net/pdf?id=SygLehCqtm

Inputs
    primary: The amino acid sequence of the protein.
    protein_length: The length of the actual sequence, used for padding and masking.
    valid_mask: Binary mask indicating which positions should not be predicted due to experimental challenges.
    evolutionary (Optional): The alignment-based features, in this case PSSMs generated from ProteinNet.


Outputs
    contact_map: Symmetric binary matrix of contacts
    contact_map_path: path to .png containing the contact map for this example.

Metadata
    id: The SCOPe ID of the protein
"""

def convert_contact_map_sequences_to_tfrecords(infile, outfile_prefix, vocab, contactmap_root, validation=False):
    sequences, names = get_contact_map_seqs(infile)
    paths = get_contact_map_paths(names, contactmap_root)

    serialize_with_vocab = partial(serialize_contact_map_sequence, vocab=vocab)

    print('Serializing...')
    with Pool() as p:
        serialized_examples = p.starmap(serialize_with_vocab, zip(sequences, names, paths))

    suffix = '_valid.tfrecords' if validation else '_train.tfrecords'
    with tf.python_io.TFRecordWriter(outfile_prefix + suffix) as writer:
        print('Creating {} ...'.format(outfile_prefix + suffix))
        for example in serialized_examples:
            writer.write(example)


def serialize_contact_map_sequence(sequence: str,
                                   name: str,
                                   path: str,
                                   vocab: Dict[str, int]):
    int_seq = []
    for aa in sequence:
        if aa in string.whitespace:
            raise ValueError("whitespace found in string")

        aa_idx = vocab.get(aa.upper())
        if aa_idx is None:
            raise ValueError(f"{aa} is not in vocab")

        int_seq.append(aa_idx)

    protein_context = to_features(
        name=name.encode('UTF-8'),
        protein_length=len(sequence),
        # needed to look up contact map later
        contact_map_path=path.encode('UTF-8'))

    protein_features = to_sequence_features(protein_sequence=int_seq)

    example = tf.train.SequenceExample(context=protein_context, feature_lists=protein_features)
    return example.SerializeToString()


def deserialize_contact_map_sequence(example):
    context = {
        'name': tf.FixedLenFeature([], tf.string),
        'protein_length': tf.FixedLenFeature([1], tf.int64),
        'contact_map_path': tf.FixedLenFeature([], tf.string)
    }

    features = {
        'protein_sequence': tf.FixedLenSequenceFeature([1], tf.int64),
    }

    context, features = tf.parse_single_sequence_example(
        example,
        context_features=context,
        sequence_features=features
    )

    name = context['name']
    protein_length = tf.cast(context['protein_length'][0], tf.int32)
    contact_map_path = context['contact_map_path']
    contact_map_raw = tf.image.decode_png(tf.io.read_file(contact_map_path))
    contact_map_raw = tf.squeeze(contact_map_raw, 2)

    contact_map = tf.cast(tf.equal(contact_map_raw, 255), tf.int32)
    valid_mask = tf.not_equal(contact_map_raw, 1)

    sequence = tf.cast(features['protein_sequence'][:, 0], tf.int32)

    return {'sequence': sequence,
            'protein_length': protein_length,
            'contact_map_path': contact_map_path,
            'contact_map': contact_map,
            'valid_mask': valid_mask,
            'name': name}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='convert protein sequences to tfrecords')
    parser.add_argument('--train', type=str, help='fasta file to convert to tfrecords')
    parser.add_argument('--validation', type=str, help='fasta file to convert to tfrecords')
    parser.add_argument('--outfile-prefix', type=str, default=None, help='name of outfile')
    parser.add_argument('--contactmap-root', type=str, default='data/SCOPe/pdbstyle-2.06/',
                        help='path to root of png contact map dir')
    args = parser.parse_args()

    vocab = PFAM_VOCAB

    convert_contact_map_sequences_to_tfrecords(args.train,
                                               args.outfile_prefix,
                                               vocab,
                                               args.contactmap_root)
    convert_contact_map_sequences_to_tfrecords(args.validation,
                                               args.outfile_prefix,
                                               vocab,
                                               args.contactmap_root,
                                               validation=True)
