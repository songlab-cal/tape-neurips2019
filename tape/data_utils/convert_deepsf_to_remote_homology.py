from typing import Dict
from pprint import pprint
from functools import partial
from multiprocessing import Pool
from glob import glob
import os

import tensorflow as tf
from Bio import SeqIO

from .remote_homology_serializer import serialize_remote_homology_sequence
from .vocabs import PFAM_VOCAB


ss_tuple_to_int = {
    (1, 0, 0): 0,
    (0, 1, 0): 1,
    (0, 0, 1): 2
}

sa_tuple_to_int = {
    (1, 0): 0,
    (0, 1): 1
}


def get_scop_labels_from_string(scop_label):
    """
    In [23]: label
    Out[23]: 'a.1.1.1'

    In [24]: get_scop_labels_from_string(label)
    Out[24]: ('a', 'a.1', 'a.1.1', 'a.1.1.1')
    """
    class_, fold, superfam, fam = scop_label.split('.')
    fold = '.'.join([class_, fold])
    superfam = '.'.join([fold, superfam])
    fam = '.'.join([superfam, fam])
    return class_, fold, superfam, fam


def get_pssm(data_dir):
    all_pssm_files = glob(os.path.join(data_dir, '*.pssm_fea'))
    pssms = {}
    with Pool() as p:
        pssm_for_scop_ids = p.map(get_pssm_for_file, all_pssm_files)
    for scop_id, pssm in pssm_for_scop_ids:
        pssms[scop_id] = pssm
    return pssms


def get_pssm_for_file(filename):
    scop_id = filename.split('/')[-1].split('.')[0]
    pssm_for_scop_id = []
    with open(filename, 'r') as f:
        lines = f.read().split()

    # 20 scores for each mutation at a given position
    position_mutations = []
    for i, line in enumerate(lines[2:]):
        if i % 20 == 0 and i != 0:
            pssm_for_scop_id.append(position_mutations)
            position_mutations = []
        mutation_score = int(line.split(':')[1])
        position_mutations.append(mutation_score)
    pssm_for_scop_id.append(position_mutations)
    return scop_id, pssm_for_scop_id


def get_sequence_and_secondary_structure_and_solvent_accessibility(data_dir):
    all_feature_files = glob(os.path.join(data_dir, '*.fea_aa_ss_sa'))
    sequence = {}
    secondary_structure = {}
    solvent_accessibility = {}

    with Pool() as p:
        extracted_data = p.map(get_seq_ss_and_sa_for_file, all_feature_files)

    for scop_id, seq, ss, sa in extracted_data:
        sequence[scop_id] = seq
        secondary_structure[scop_id] = ss
        solvent_accessibility[scop_id] = sa

    return sequence, secondary_structure, solvent_accessibility


def get_seq_ss_and_sa_for_file(filename):
    scop_id = filename.split('/')[-1].split('.')[0]
    sequence = []
    secondary_structure = []
    solvent_accessibility = []
    with open(filename, 'r') as f:
        lines = f.read().split()

    # clip the beginning
    lines = lines[2:]
    # split off indexes
    bits = [int(l.split(':')[1]) for l in lines]

    # 20 amino acids 1 hot
    # Followed by 3 secondary structure labels
    # Followed by 2 solvent accessibility labels
    aa_label = []
    ss_label = []
    sa_label = []
    for i, bit in enumerate(bits, start=0):
        if i % 25 == 0 and i != 0:
            # sequence.append(aa_tuple_to_int[tuple(aa_label)])
            assert len(aa_label) <= 20
            assert len(ss_label) <= 3
            assert len(sa_label) <= 2
            sequence.append(tuple(aa_label))
            secondary_structure.append(ss_tuple_to_int[tuple(ss_label)])
            solvent_accessibility.append(sa_tuple_to_int[tuple(sa_label)])
            aa_label = []
            ss_label = []
            sa_label = []
        if i % 25 < 20:
            aa_label.append(bit)
        elif i % 25 >= 20 and i % 25 < 23:
            ss_label.append(bit)
        else:
            sa_label.append(bit)
    sequence.append(tuple(aa_label))
    secondary_structure.append(ss_tuple_to_int[tuple(ss_label)])
    solvent_accessibility.append(sa_tuple_to_int[tuple(sa_label)])
    return scop_id, sequence, secondary_structure, solvent_accessibility


def get_scope_id_to_label(filename):
    scop_id_to_label = {}
    with open(filename, 'r') as f:
        for l in f:
            scop_id = l.split()[0]
            label = l.split()[2]
            scop_id_to_label[scop_id] = label
    return scop_id_to_label


def convert_deepsf_data_to_tfrecords(filenames: str,
                                     outfilenames: str,
                                     feature_dir: str,
                                     pssm_dir: str,
                                     fasta: str,
                                     vocab: Dict[str, int]):
    serialize_with_vocab = partial(serialize_remote_homology_sequence, vocab=vocab)

    # need to construct dictionaries with label string:int maps
    class_to_int_label = {}
    fold_to_int_label = {}
    superfamily_to_int_label = {}
    family_to_int_label = {}

    seq_dict, ss_dict, sa_dict = get_sequence_and_secondary_structure_and_solvent_accessibility(feature_dir)
    pssm_dict = get_pssm(pssm_dir)
    # I don't trust the sequences in the deepsf feature files, so we will look them up in the fasta
    scop_id_to_scop_string = {record.name: record.description.split()[1] for record in SeqIO.parse(fasta, 'fasta')}
    scop_id_to_seq = {record.name: str(record.seq).upper() for record in SeqIO.parse(fasta, 'fasta')}

    skipped_ids = []
    incongruent_ids = []
    for filename, outfilename in zip(filenames, outfilenames):
        to_tfrecord = []
        scop_id_to_label = get_scope_id_to_label(filename)

        for scop_id, _ in scop_id_to_label.items():
            use_alt = False
            alt = scop_id.replace('_', '.')
            if scop_id not in scop_id_to_seq:
                if alt not in scop_id_to_seq:
                    skipped_ids.append(scop_id)
                    continue
                else:
                    use_alt = True
            if use_alt:
                seq = scop_id_to_seq[alt]
                scop_string = scop_id_to_scop_string[alt]
            else:
                seq = scop_id_to_seq[scop_id]
                scop_string = scop_id_to_scop_string[scop_id]
            seq_id = scop_id
            class_, fold, superfam, fam = get_scop_labels_from_string(scop_string)
            class_label = class_to_int_label.setdefault(class_, len(class_to_int_label))
            fold_label = fold_to_int_label.setdefault(fold, len(fold_to_int_label))
            superfamily_label = superfamily_to_int_label.setdefault(superfam, len(superfamily_to_int_label))
            family_label = family_to_int_label.setdefault(fam, len(family_to_int_label))

            pssm = pssm_dict[seq_id]
            ss = ss_dict[seq_id]
            sa = sa_dict[seq_id]
            if not all([len(pssm) == len(ss), len(ss) == len(sa), len(sa) == len(seq)]):
                print('id {} pssm {} ss {} sa {} parsed aa {} fasta {}'.format(seq_id,
                                                                               len(pssm),
                                                                               len(ss),
                                                                               len(sa),
                                                                               len(seq_dict[seq_id]),
                                                                               len(seq)))
                incongruent_ids.append(scop_id)
                continue
            # assert(len(pssm) == len(ss) == len(sa) == len(seq))
            to_tfrecord.append([seq, seq_id, class_label, fold_label, superfamily_label, family_label, pssm, ss, sa])

        print('Serializing {} examples...'.format(len(to_tfrecord)))
        with Pool() as p:
            serialized_examples = p.starmap(serialize_with_vocab, to_tfrecord)

        with tf.python_io.TFRecordWriter(outfilename) as writer:
            print('Creating TFrecords...')
            for serialized_example in serialized_examples:
                writer.write(serialized_example)

    print('Incongruent {}'.format(len(incongruent_ids)))
    print('Skipped {}'.format(len(skipped_ids)))
    pprint(incongruent_ids)
    pprint(skipped_ids)
    print('Num classes {}'.format(len(class_to_int_label)))
    pprint(class_to_int_label)
    print('Num folds {}'.format(len(fold_to_int_label)))
    pprint(fold_to_int_label)
    print('Num superfams {}'.format(len(superfamily_to_int_label)))
    pprint(superfamily_to_int_label)
    print('Num fams {}'.format(len(family_to_int_label)))
    pprint(family_to_int_label)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='convert protein sequences to tfrecords')
    parser.add_argument('--listdir', required=True, help='DeepSF train split list directory: each list contains scop ids')
    parser.add_argument('--outprefix', required=True, help='prefix for output files')
    parser.add_argument('--featuredir', required=True)
    parser.add_argument('--pssmdir', required=True)
    parser.add_argument('--fasta', required=True)
    args = parser.parse_args()

    vocab = PFAM_VOCAB

    filenames = glob(os.path.join(args.listdir, '*.list*'))

    infilenames = ['test_dataset.list_family', 'test_dataset.list_superfamily', 'test_dataset.list_fold', 'Traindata.list',  'validation.list']
    infilenames = [os.path.join(args.listdir, name) for name in infilenames]

    outfilenames = ['remote_homology_test_family_holdout.tfrecords',
                    'remote_homology_test_superfamily_holdout.tfrecords',
                    'remote_homology_test_fold_holdout.tfrecords',
                    'remote_homology_valid.tfrecords',
                    'remote_homology_train.tfrecords']
    outfilenames = [os.path.join(args.outprefix, name) for name in outfilenames]

    convert_deepsf_data_to_tfrecords(infilenames,
                                     outfilenames,
                                     args.featuredir,
                                     args.pssmdir,
                                     args.fasta,
                                     vocab)


# all deepsf data can be downloaded from
# http://iris.rnet.missouri.edu/DeepSF/download.html

# fasta downloaded from
# http://iris.rnet.missouri.edu/DeepSF/download/PDB_SCOP95_seq_scop1.75.txt

# DONT USE!!
# astral scop was downloaded from
# https://scop.berkeley.edu/downloads/scopseq-1.75/astral-scopdom-seqres-all-1.75.fa


# python -m tape.data_utils.convert_deepsf_to_remote_homology \
#     --listdir ~/git/DeepSF/datasets/D2_Three_levels_dataset/ \
#     --outprefix data/remote_homology_tfrecord/ \
#     --featuredir ~/git/DeepSF/datasets/features/Feature_aa_ss_sa/ \
#     --pssmdir ~/git/DeepSF/datasets/features/PSSM_Fea/ \
#     --fasta ~/git/DeepSF/PDB_SCOP95_seq_scop1.75.txt > out.txt
