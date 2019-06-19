from typing import Dict, Optional, Tuple

import string
import tensorflow as tf
from tqdm import tqdm
from Bio import SeqIO
import pandas as pd
from Bio.SeqRecord import SeqRecord

import pickle as pkl


from .tf_data_utils import to_features, to_sequence_features

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


def parse_line(record: SeqRecord, fam_clan_dict: Dict[str, str]) -> Tuple[str, str, str]:
    seq = record.seq
    family = record.description.split(' ')[2].split('.')[0]  # Yeah, it's ugly

    # Some families don't even show up?
    clan = fam_clan_dict.get(family)
    if clan is None:
        clan = 'no_clan'

    return seq, family, clan


def convert_pfam_sequences_to_tfrecords(filename: str,
                                        outfile: Optional[str],
                                        fam_clan_dict: Dict[str, str],
                                        vocab: Optional[Dict[str, int]] = None) -> None:
    if outfile is None:
        outfile = filename.rsplit('.')[0]
    else:
        outfile = outfile.rsplit('.')[0]

    if vocab is None:
        vocab = {"<PAD>": 0, "<MASK>": 1, "<CLS>": 2, "<SEP>": 3}

    fam_dict: Dict[str, int] = {}
    clan_dict = {'no_clan': 0}

    print("Forming Train Set")

    with tf.python_io.TFRecordWriter(outfile + '.tfrecords') as writer:
        for record in tqdm(SeqIO.parse(filename, 'fasta'), total=34353433):
            seq, family, clan = parse_line(record, fam_clan_dict)

            serialized_example, vocab, fam_dict, clan_dict = serialize_pfam_sequence(
                seq, family, clan, vocab, fam_dict, clan_dict)

            writer.write(serialized_example)

    print("Dumping Dicts")
    with open(outfile.rsplit('.', maxsplit=1)[0] + '.vocab', 'wb') as f:
        pkl.dump(vocab, f)

    with open(outfile.rsplit('.', maxsplit=1)[0] + '_fams.pkl', 'wb') as f:
        pkl.dump(fam_dict, f)

    with open(outfile.rsplit('.', maxsplit=1)[0] + '_clans.pkl', 'wb') as f:
        pkl.dump(clan_dict, f)


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
        if aa_idx is None:
            vocab[aa] = len(vocab)  # Can't do this with defaultdict b/c it depends on the dictionary
            aa_idx = vocab[aa]

        int_sequence.append(aa_idx)

    clan_idx = clan_dict.get(clan)
    if clan_idx is None:
        clan_dict[clan] = len(clan_dict)
        clan_idx = clan_dict[clan]

    fam_idx = fam_dict.get(family)
    if fam_idx is None:
        fam_dict[family] = len(fam_dict)
        fam_idx = fam_dict[family]

    protein_context = to_features(protein_length=len(int_sequence), clan=clan_idx, family=fam_idx)
    protein_features = to_sequence_features(sequence=int_sequence)

    example = tf.train.SequenceExample(context=protein_context, feature_lists=protein_features)
    return example.SerializeToString(), vocab, fam_dict, clan_dict


def deserialize_pfam_sequence(example, add_cls_token: bool = False, cls_token: Optional[int] = None):
    assert not (add_cls_token and cls_token is None)
    context = {
        'protein_length': tf.FixedLenFeature([1], tf.int64),
        'clan': tf.FixedLenFeature([1], tf.int64),
        'family': tf.FixedLenFeature([1], tf.int64)
    }

    features = {
        'sequence': tf.FixedLenSequenceFeature([1], tf.int64),
    }

    context, features = tf.parse_single_sequence_example(
        example,
        context_features=context,
        sequence_features=features
    )

    protein_length = tf.to_int32(context['protein_length'][0])
    clan = tf.cast(context['clan'][0], tf.int32)
    family = tf.cast(context['family'][0], tf.int32)
    sequence = tf.to_int32(features['sequence'][:, 0])

    if add_cls_token:
        sequence = tf.pad(sequence, [[1, 0]], constant_values=cls_token)
        protein_length += 1

    return {'serialized': example,
            'sequence': sequence,
            'protein_length': protein_length,
            'clan': clan,
            'family': family}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='convert protein sequences to tfrecords')
    parser.add_argument('filename', type=str, help='text file to convert to tfrecords')
    parser.add_argument('--fam-to-clan-file', type=str, help='.TSV file mapping families to clans')
    parser.add_argument('--outfile', type=str, default=None, help='name of outfile')
    args = parser.parse_args()

    FAM_TO_CLAN_MAP = form_clan_fam_map(args.fam_to_clan_file)
    print("Family to Clan dict Loaded")
    convert_pfam_sequences_to_tfrecords(args.filename, args.outfile, FAM_TO_CLAN_MAP)
