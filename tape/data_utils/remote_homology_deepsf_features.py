import glob
import os
from multiprocessing import Pool

deepsf_dir = os.path.expanduser('~/git/DeepSF')
pssm_dir = os.path.join(deepsf_dir, 'datasets/features/PSSM_Fea/')
secondary_structure_and_solvent_accessibility_dir = os.path.join(deepsf_dir, 'datasets/features/Feature_aa_ss_sa/')


def get_pssm_features(data_dir):
    all_pssm_files = glob.glob(os.path.join(data_dir, '*.pssm_fea'))
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
        mutation_score = line.split(':')[1]
        position_mutations.append(mutation_score)
    return scop_id, pssm_for_scop_id


def get_secondary_structure_and_solvent_accessibility(data_dir):
    all_feature_files = glob.glob(os.path.join(data_dir, '*.fea_aa_ss_sa'))
    secondary_structure = {}
    solvent_accessibility = {}
    with Pool() as p:
        ss_and_sa_for_scop_ids = p.map(get_ss_and_sa_for_file, all_feature_files)
    for scop_id, ss, sa in ss_and_sa_for_scop_ids:
        secondary_structure[scop_id] = ss
        solvent_accessibility[scop_id] = sa

    return secondary_structure, solvent_accessibility


ss_tuple_to_int = {
    (1, 0, 0): 0,
    (0, 1, 0): 1,
    (0, 0, 1): 2
}

sa_tuple_to_int = {
    (1, 0): 0,
    (0, 1): 1
}


def get_ss_and_sa_for_file(filename):
    scop_id = filename.split('/')[-1].split('.')[0]
    ss = []
    sa = []
    with open(filename, 'r') as f:
        lines = f.read().split()

    # 20 amino acids 1 hot
    # Followed by 3 secondary structure labels
    # Followed by 2 solvent accessibility labels
    ss_label = []
    sa_label = []
    for i, line in enumerate(lines[2:]):
        if i % 25 == 0 and i != 0:
            ss.append(ss_tuple_to_int[tuple(ss_label)])
            sa.append(sa_tuple_to_int[tuple(sa_label)])
            ss_label = []
            sa_label = []
            continue
        if i % 25 < 20:
            continue
        elif i % 25 % 20 < 3:
            ss_label.append(int(line.split(':')[1]))
        else:
            sa_label.append(int(line.split(':')[1]))
    return scop_id, ss, sa


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Get PSSMs from DeepSF')
    parser.add_argument('--datadir', type=str, default=None, help='where the pssms are')
    args = parser.parse_args()
    pssms = get_pssm_features(args.datadir)
