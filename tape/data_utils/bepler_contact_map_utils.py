from __future__ import print_function, division

import numpy as np
import os
import glob
from PIL import Image

from Bio import SeqIO


astral_train_path = 'data/SCOPe/astral-scopedom-seqres-gd-sel-gs-bib-95-2.06.train.fa'
astral_test_path = 'data/SCOPe/astral-scopedom-seqres-gd-sel-gs-bib-95-2.06.test.fa'


def get_contact_map_seqs(astral_fasta_path):
    sequences = []
    names = []
    for record in SeqIO.parse(astral_fasta_path, 'fasta'):
        sequences.append(str(record.seq).upper())
        names.append(record.name)
    return sequences, names


def get_contact_map_paths(names, images_root='data/SCOPe/pdbstyle-2.06/'):
    cmap_paths = glob.glob(images_root + '*/*.png')
    cmap_dict = {os.path.basename(path)[:7]: path for path in cmap_paths}

    paths = []
    for name in names:
        if name not in cmap_dict:
            name = 'd' + name[1:]
        path = cmap_dict[name]
        paths.append(path)
    return paths


def get_binary_contact_map(path):
    """
    For use in the crossentropy loss
    """
    im = np.array(Image.open(path), copy=False)
    contacts = np.zeros(im.shape, dtype=np.int32)
    # what does this pixel intensity=1 mean?
    contacts[im == 1] = -1
    contacts[im == 255] = 1
    # mask the matrix below the diagonal
    mask = np.tril_indices(contacts.shape[0], k=-1)
    contacts[mask] = -1
    return contacts


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='convert protein sequences to tfrecords')
    parser.add_argument('astral', type=str, help='fasta file to convert to tfrecords')
    args = parser.parse_args()

    get_contact_map_paths(args.astral)
