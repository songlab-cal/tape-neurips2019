from typing import Dict, Optional, Tuple
import os
import tensorflow as tf
import numpy as np

from tqdm import tqdm

from .tf_data_utils import to_features, to_sequence_features
from .vocabs import FLIPPED_AA, PFAM_VOCAB, SS_8_TO_3_ENCODED

"""
Inputs
    primary: The amino acid sequence of the protein.
    protein_length: The length of the actual sequence, used for padding and masking.
    valid_mask: Binary mask indicating which positions should not be predicted due to experimental challenges.
    evolutionary (Optional): The alignment-based features, in this produced by HHblits.

Outputs

You can predict either 3-way or 8-way secondary structure: 

    ss3: The secondary structure at each position, given by a 3-way label.
    ss8: The secondary structure at each position, given by a 8-way label.

NetSurfP2.0 performs multitask training on the following optional labels

    rsa (Optional): Relative solvent accessibility at each position, a scalar. 
    phi (Optional): A bond angle at each position, returned as a vector [cos(x), sin(x)]. 
    psi (Optional): A second bond angle at each position, returned as a vector [cos(x), sin(x)].
    disorder (Optional): A binary label denoting ordered or disordered at each position.

Metadata
    id: The id of the protein, drawn from PDB.
    asa_max: The absolute maximum solvent accessibility, used to rescale relative solvent accessibility to absolute solvent accessibility for some analyses.
    interface: A binary label saying whether a given protein is at the boundary of two chains in a multi-chain protein.

"""

def read_secondary_structure_data(filename: str) -> Dict[str, np.array]:
    """
    Takes in file to .npz file containing secondary structure data. Returns sequences,
    sequence lengths, secondary structure tags, and PDB ids.

    Assumes data comes as a 3-d array of sample x sequence x datatype, as found in the
    NetSurfP2.0 paper described here http://www.cbs.dtu.dk/services/NetSurfP/
    All datasets are the HHblits version.
    """

    archive = np.load(filename)

    pdb_ids = archive['pdbids']
    data = archive['data']

    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    pdb_ids = pdb_ids[indices]

    # Use link in comment above to get out data
    # [0:20] Amino Acids (sparse encoding)
    # Unknown residues are stored as an all-zero vector
    seqs = data[:, :, 0:20]
    # [20:50] hmm profile
    hmm_profile = data[:, :, 20:50]
    # [50] Seq mask (1 = seq, 0 = empty)
    sequence_masks = data[:, :, 50]
    # [51] Disordered mask (0 = disordered, 1 = ordered)
    disorder = data[:, :, 51]
    # [52] Evaluation mask (For CB513 dataset, 1 = eval, 0 = ignore)
    valid_mask = data[:, :, 52]
    # [53] ASA (isolated)
    asa_iso = data[:, :, 53]
    # [54] ASA (complexed)
    asa_com = data[:, :, 54]
    # [55] RSA (isolated)
    rsa_iso = data[:, :, 55]
    # [56] RSA (complexed)
    rsa_com = data[:, :, 56]  # noqa: F841
    # [57:65] Q8 GHIBESTC (Q8 -> Q3: HHHEECCC)
    secondary_structure = data[:, :, 57:65]
    # [65:67] Phi+Psi
    phi = data[:, :, 65]
    psi = data[:, :, 66]
    # [67] ASA_max
    asa_max = data[:, :, 67]

    interface = np.asarray(np.abs(asa_iso - asa_com) > 1, np.int32)

    # Convert one hot to indexed for sequences
    seqs = seqs * np.arange(1, 21)  # encode unknown as -1
    seqs = np.sum(seqs, axis=2) - 1

    # Convert one hot to indexed for secondary structure
    secondary_structure = secondary_structure * np.arange(8)
    secondary_structure = np.sum(secondary_structure, axis=2)

    # Extract lengths
    seq_lengths = np.sum(sequence_masks, axis=1).astype(np.int64)

    return {
        'primary': seqs,
        'evolutionary': hmm_profile,
        'protein_length': seq_lengths,
        'valid_mask': valid_mask,
        'secondary_structure': secondary_structure,
        'disorder': disorder,
        'phi': phi,
        'psi': psi,
        'asa_max': asa_max,
        'rsa': rsa_iso,
        'id': pdb_ids,
        'interface': interface}


def write_file(filename, dataset):
    print('Writing {}...'.format(filename))
    with tf.python_io.TFRecordWriter(filename) as writer:
        for vals in tqdm(zip(*dataset.values())):
            itemdict = dict(zip(dataset.keys(), vals))
            serialized_example = serialize_secondary_structure_sequence(itemdict)
            writer.write(serialized_example)


def convert_secondary_structure_data_to_tfrecords(vocab: Dict[str, int],
                                       outfile: Optional[str],
                                       directory: str) -> None:
    print("Reading Data")
    train_file = os.path.join(directory, 'Train_HHblits.npz')
    casp_file = os.path.join(directory, 'CASP12_HHblits.npz')
    ts_file = os.path.join(directory, 'TS115_HHblits.npz')
    cb_file = os.path.join(directory, 'CB513_HHblits.npz')

    # Train validation split of training data
    data = read_secondary_structure_data(train_file)
    train_split = int(0.8 * data['primary'].shape[0])
    train_data = {key: array[:train_split] for key, array in data.items()}
    valid_data = {key: array[train_split:] for key, array in data.items()}

    # Get fancier sets
    casp_data = read_secondary_structure_data(casp_file)
    ts_data = read_secondary_structure_data(ts_file)
    cb_data = read_secondary_structure_data(cb_file)

    write_file('secondary_structure_train.tfrecords', train_data)
    write_file('secondary_structure_valid.tfrecords', valid_data)
    write_file('secondary_structure_casp12.tfrecords', casp_data)
    write_file('secondary_structure_ts115.tfrecords', ts_data)
    write_file('secondary_structure_cb513.tfrecords', cb_data)


def to_three(seq):
    new_seq = np.zeros_like(seq)
    for eight_encoding, three_encoding in SS_8_TO_3_ENCODED.items():
        new_seq[seq == eight_encoding] = three_encoding
    return new_seq


def serialize_secondary_structure_sequence(example) -> bytes:
    protein_length = example['protein_length']
    # Raw sequence
    sequence = example['primary'][:protein_length]
    sequence = sequence.astype(np.int32).tolist()
    sequence = [vocab[FLIPPED_AA[position]] for position in sequence]

    # Profile features
    profile = example['evolutionary'][:protein_length, :].astype(np.float32)

    # Mask
    valid_mask = example['valid_mask'][:protein_length]

    # Secondary structure labels
    ss_8 = example['secondary_structure'][:protein_length]
    ss_3 = to_three(ss_8)
    ss_8 = ss_8.astype(np.int32).tolist()
    ss_3 = ss_3.astype(np.int32).tolist()

    # Disorder labels
    disorder = example['disorder'][:protein_length]
    disorder = disorder.astype(np.int32).tolist()

    # Phi and psi labels
    psi = example['psi'][:protein_length]
    phi = example['phi'][:protein_length]

    # Solvent accessibility labels
    asa_max = example['asa_max'][:protein_length]
    rsa = example['rsa'][:protein_length]

    # Interface
    interface = example['interface'][:protein_length]

    # Other features
    protein_length = int(protein_length)
    pdb_id = example['id'].encode('UTF-8')

    protein_context = to_features(
        id=pdb_id,
        protein_length=protein_length)
    protein_features = to_sequence_features(
        primary=sequence,
        evolutionary=profile,
        valid_mask=valid_mask,
        ss8=ss_8,
        ss3=ss_3,
        disorder=disorder,
        phi=phi,
        psi=psi,
        rsa=rsa,
        interface=interface,
        asa_max=asa_max)

    example = tf.train.SequenceExample(context=protein_context, feature_lists=protein_features)
    return example.SerializeToString()

def deserialize_secondary_structure(example):
    context = {
        'id': tf.FixedLenFeature([], tf.string),
        'protein_length': tf.FixedLenFeature([], tf.int64)
    }

    features = {
        'primary': tf.FixedLenSequenceFeature([], tf.int64),
        'evolutionary': tf.FixedLenSequenceFeature([30], tf.float32),
        'ss3': tf.FixedLenSequenceFeature([], tf.int64),
        'ss8': tf.FixedLenSequenceFeature([], tf.int64),
        'disorder': tf.FixedLenSequenceFeature([], tf.int64),
        'interface': tf.FixedLenSequenceFeature([], tf.int64),
        'phi': tf.FixedLenSequenceFeature([], tf.float32),
        'psi': tf.FixedLenSequenceFeature([], tf.float32),
        'rsa': tf.FixedLenSequenceFeature([], tf.float32),
        'asa_max': tf.FixedLenSequenceFeature([], tf.float32),
        'valid_mask': tf.FixedLenSequenceFeature([], tf.float32)
    }

    context, features = tf.parse_single_sequence_example(
        example,
        context_features=context,
        sequence_features=features
    )

    features.update(context)

    for name, feature in features.items():
        if feature.dtype == tf.int64:
            features[name] = tf.cast(feature, tf.int32)

    return features


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='convert protein sequences to tfrecords')
    parser.add_argument('directory', type=str, help='text file to convert to tfrecords')
    parser.add_argument('--outfile', type=str, default=None, help='name of outfile')
    args = parser.parse_args()

    vocab = PFAM_VOCAB
    convert_secondary_structure_data_to_tfrecords(vocab, args.outfile, args.directory)
