from typing import Dict, List
import string

import tensorflow as tf

from .tf_data_utils import to_features, to_sequence_features

"""
Inputs
    primary: The amino acid sequence of the protein.
    protein_length: The length of the actual sequence, used for padding and masking.
    evolutionary (Optional): The alignment-based features, in this case PSSMs.
    secondary_structure (Optional): A secondary structure 3-class label at each position.
    solvent_accessibility (Optional): A solvent accessibility 2-class label at each position.

Outputs
    fold_label: The protein fold, an 1195-class label for the whole protein.

Metadata
    class_label: The protein class, an 8-class label that is coarser than fold.
    superfamily_label: The protein superfamily, a finer label than fold.
    family_label: The protein family, a finer label than superfamily.
    id: The id of the protein, drawn from SCOPe.

"""


def serialize_remote_homology_sequence(sequence: str,
                                       seq_id: str,
                                       class_label: int,
                                       fold_label: int,
                                       superfamily_label: int,
                                       family_label: int,
                                       pssm: List[List[int]],
                                       secondary_structure: List[int],
                                       solvent_accessibility: List[int],
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
        class_label=class_label,
        fold_label=fold_label,
        superfamily_label=superfamily_label,
        family_label=family_label)

    protein_features = to_sequence_features(primary=int_sequence,
                                            secondary_structure=secondary_structure,
                                            solvent_accessibility=solvent_accessibility,
                                            evolutionary=pssm)

    example = tf.train.SequenceExample(context=protein_context, feature_lists=protein_features)
    return example.SerializeToString()


def deserialize_remote_homology_sequence(example):
    context = {
        'id': tf.FixedLenFeature([], tf.string),
        'protein_length': tf.FixedLenFeature([], tf.int64),
        'class_label': tf.FixedLenFeature([], tf.int64),
        'fold_label': tf.FixedLenFeature([], tf.int64),
        'superfamily_label': tf.FixedLenFeature([], tf.int64),
        'family_label': tf.FixedLenFeature([], tf.int64)
    }

    features = {
        'primary': tf.FixedLenSequenceFeature([], tf.int64),
        'secondary_structure': tf.FixedLenSequenceFeature([], tf.int64),
        'solvent_accessibility': tf.FixedLenSequenceFeature([], tf.int64),
        'evolutionary': tf.FixedLenSequenceFeature([20], tf.int64)
    }

    context, features = tf.parse_single_sequence_example(
        example,
        context_features=context,
        sequence_features=features
    )

    secondary_structure = tf.one_hot(features['secondary_structure'], 3)
    solvent_accessibility = tf.one_hot(features['solvent_accessibility'], 2)
    # floats since that's what downstream model expects
    evolutionary = tf.cast(features['evolutionary'], tf.float32)

    return {'id': context['id'],
            'primary': tf.to_int32(features['primary']),
            'protein_length': tf.to_int32(context['protein_length']),
            'class_label': tf.cast(context['class_label'], tf.int32),
            'fold_label': tf.cast(context['fold_label'], tf.int32),
            'superfamily_label': tf.cast(context['superfamily_label'], tf.int32),
            'family_label': tf.cast(context['family_label'], tf.int32),
            'secondary_structure': secondary_structure,
            'solvent_accessibility': solvent_accessibility,
            'evolutionary': evolutionary}
