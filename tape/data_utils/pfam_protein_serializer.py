from typing import Dict, Optional, Tuple

import string
import pickle as pkl
import random
from functools import partial
from multiprocessing import Pool

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import tensorflow as tf
from tqdm import tqdm
import pandas as pd

from .tf_data_utils import to_features, to_sequence_features
from .vocabs import PFAM_VOCAB

"""
Inputs
    primary: The amino acid sequence of the protein.
    protein_length: The length of the actual sequence, used for padding and masking.

Outputs
    None

Metadata
    family: Functional/Evolutionary label provided for all protein domains in Pfam.
    clan: Higher-level label provided for some protein domains in Pfam.
"""


def form_clan_fam_map(clan_fam_file: str) -> Dict[str, str]:
    data = pd.read_csv(clan_fam_file, sep='\t', na_values='\t')
    data = data.fillna('no_clan')  # Replace nans with simple string
    families = list(data.iloc[:, 0])
    clans = list(data.iloc[:, 1])
    return dict(zip(families, clans))


def parse_line(record: SeqRecord, fam_to_clan_dict: Dict[str, str]) -> Tuple[str, str, str]:
    seq = record.seq
    family = record.description.split(' ')[2].split('.')[0]  # Yeah, it's ugly

    # Some families don't even show up?
    clan = fam_to_clan_dict.get(family)
    if clan is None:
        clan = 'no_clan'

    return seq, family, clan


def convert_pfam_sequences_to_tfrecords(filename: str,
                                        outfile: Optional[str],
                                        fam_to_clan_dict: Dict[str, str],
                                        fam_dict: Dict[str, int],
                                        clan_dict: Dict[str, int],
                                        seed: int = 0,
                                        vocab: Optional[Dict[str, int]] = None) -> None:
    if outfile is None:
        outfile = filename.rsplit('.')[0]
    else:
        outfile = outfile.rsplit('.')[0]

    if vocab is None:
        vocab = {"<PAD>": 0, "<MASK>": 1, "<CLS>": 2, "<SEP>": 3}

    serialize_map_fn = partial(serialize_pfam_sequence, vocab=vocab, fam_dict=fam_dict, clan_dict=clan_dict)
    print("Forming Train Set")

    all_examples = []
    holdout_examples = []
    holdout_clans = ['CL0635', 'CL0624', 'CL0355', 'CL0100', 'CL0417', 'CL0630']
    holdout_families = ['PF18346', 'PF14604', 'PF18697', 'PF03577', 'PF01112', 'PF03417']

    for record in tqdm(SeqIO.parse(filename, 'fasta'), total=34353433):
        seq, family, clan = parse_line(record, fam_to_clan_dict)
        if clan in holdout_clans or family in holdout_families:
            holdout_examples.append((seq, family, clan))
        else:
            all_examples.append((seq, family, clan))

    print("Shuffling")
    random.seed(seed)
    random.shuffle(all_examples)

    print("Writing holdout")
    with tf.python_io.TFRecordWriter(outfile + '_holdout' + '.tfrecords') as writer:
        for seq, family, clan in holdout_examples:
            serialized_example = serialize_pfam_sequence(seq, family, clan, vocab, fam_dict, clan_dict)
            writer.write(serialized_example)

    num_files = 60
    print("Serializing training examples")
    with Pool() as p:
        serialized_examples = p.starmap(serialize_map_fn, all_examples)

    print("Writing training set")
    for i in range(num_files):
        filename = outfile + '_' + str(i) + '.tfrecords'
        with tf.python_io.TFRecordWriter(filename) as writer:
            for serialized_example in serialized_examples[i::num_files]:
                writer.write(serialized_example)


def serialize_pfam_sequence(sequence: str,
                            family: str,
                            clan: str,
                            vocab: Dict[str, int],
                            fam_dict: Dict[str, int],
                            clan_dict: Dict[str, int]) -> Tuple[bytes, Dict[str, int], Dict[str, int], Dict[str, int]]:
    int_sequence = []
    for aa in sequence:
        if aa in string.whitespace:
            raise ValueError("whitespace found in string")

        aa_idx = vocab.get(aa)
        int_sequence.append(aa_idx)

    clan_idx = clan_dict.get(clan)
    fam_idx = fam_dict.get(family)

    protein_context = to_features(protein_length=len(int_sequence), clan=clan_idx, family=fam_idx)
    protein_features = to_sequence_features(primary=int_sequence)

    example = tf.train.SequenceExample(context=protein_context, feature_lists=protein_features)
    return example.SerializeToString()


def deserialize_pfam_sequence(example):
    context = {
        'protein_length': tf.FixedLenFeature([1], tf.int64),
        'clan': tf.FixedLenFeature([1], tf.int64),
        'family': tf.FixedLenFeature([1], tf.int64)
    }

    features = {
        'primary': tf.FixedLenSequenceFeature([1], tf.int64),
    }

    context, features = tf.parse_single_sequence_example(
        example,
        context_features=context,
        sequence_features=features
    )

    protein_length = tf.to_int32(context['protein_length'][0])
    clan = tf.cast(context['clan'][0], tf.int32)
    family = tf.cast(context['family'][0], tf.int32)
    primary = tf.to_int32(features['primary'][:, 0])

    return {'primary': primary,
            'protein_length': protein_length,
            'clan': clan,
            'family': family}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='convert protein sequences to tfrecords')
    parser.add_argument('filename', type=str, help='text file to convert to tfrecords')
    parser.add_argument('--fam-to-clan-file', type=str, help='.TSV file mapping families to clans')
    parser.add_argument('--fampkl', type=str, help='python pickle mapping families to ints')
    parser.add_argument('--clanpkl', type=str, help='python pickle mapping clans to ints')
    parser.add_argument('--outfile', type=str, default=None, help='name of outfile')
    args = parser.parse_args()

    fam_to_clan_dict = form_clan_fam_map(args.fam_to_clan_file)
    with open(args.fampkl, 'rb') as f:
        fam_to_int_dict = pkl.load(f)
    with open(args.clanpkl, 'rb') as f:
        clan_to_int_dict = pkl.load(f)
    convert_pfam_sequences_to_tfrecords(args.filename,
                                        args.outfile,
                                        fam_to_clan_dict,
                                        fam_to_int_dict,
                                        clan_to_int_dict,
                                        vocab=PFAM_VOCAB)

# python -m tape.data_utils.pfam_protein_serializer \
#     data/pfam/Pfam-A.fasta \
#     --outfile data/pfam_reserialize/pfam31 \
#     --fam-to-clan-file data/pfam/Pfam-A.clans.tsv \
#     --fampkl data/pkls/pfam_fams_public.pkl \
#     --clanpkl data/pkls/pfam_clans_public.pkl
