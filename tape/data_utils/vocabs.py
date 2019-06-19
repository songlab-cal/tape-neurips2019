PFAM_VOCAB = {
    '<PAD>': 0,
    '<MASK>': 1,
    '<CLS>': 2,
    '<SEP>': 3,
    'A': 4,
    'B': 5,
    'C': 6,
    'D': 7,
    'E': 8,
    'F': 9,
    'G': 10,
    'H': 11,
    'I': 12,
    'K': 13,
    'L': 14,
    'M': 15,
    'N': 16,
    'O': 17,
    'P': 18,
    'Q': 19,
    'R': 20,
    'S': 21,
    'T': 22,
    'U': 23,
    'V': 24,
    'W': 25,
    'X': 26,
    'Y': 27,
    'Z': 28}

AA_DICT = {
    'X': -1,  # unknown
    'A': 0,
    'C': 1,
    'D': 2,
    'E': 3,
    'F': 4,
    'G': 5,
    'H': 6,
    'I': 7,
    'K': 8,
    'L': 9,
    'M': 10,
    'N': 11,
    'P': 12,
    'Q': 13,
    'R': 14,
    'S': 15,
    'T': 16,
    'V': 17,
    'W': 18,
    'Y': 19}

FLIPPED_AA = {v: k for k, v in AA_DICT.items()}

UNIPROT_BEPLER = {
    'A': 0,
    'R': 1,
    'N': 2,
    'D': 3,
    'C': 4,
    'Q': 5,
    'E': 6,
    'G': 7,
    'H': 8,
    'I': 9,
    'L': 10,
    'K': 11,
    'M': 12,
    'F': 13,
    'P': 14,
    'S': 15,
    'T': 16,
    'W': 17,
    'Y': 18,
    'V': 19,
    'X': 20,
    'O': 11,
    'U': 4,
    'B': 20,
    'Z': 20}

UNIREP_VOCAB = {
    '<PAD>': 0,
    'M': 1,
    'R': 2,
    'H': 3,
    'K': 4,
    'D': 5,
    'E': 6,
    'S': 7,
    'T': 8,
    'N': 9,
    'Q': 10,
    'C': 11,
    'U': 12,
    'G': 13,
    'P': 14,
    'A': 15,
    'V': 16,
    'I': 17,
    'F': 18,
    'Y': 19,
    'W': 20,
    'L': 21,
    'O': 22,  # Pyrrolysine
    'X': 23,  # Unknown
    'Z': 23,  # Glutamic acid or GLutamine
    'B': 23,  # Asparagine or aspartic acid
    'J': 23,  # Leucine or isoleucine
    '<START>': 24,
    '<STOP>': 25,
}

SS_8_DICT = {
           0: 'G',    # 3-turn helix (310 helix). Min length 3 residues.
           1: 'H',    # 4-turn helix (α helix). Minimum length 4 residues.
           2: 'I',    # 5-turn helix (π helix). Minimum length 5 residues.
           3: 'B',    # residue in isolated β-bridge (single pair β-sheet hydrogen bond formation)
           4: 'E',    # extended strand in parallel and/or anti-parallel β-sheet conformation. Min length 2 residues.
           5: 'S',    # bend (the only non-hydrogen-bond based assignment).
           6: 'T',    # hydrogen bonded turn (3, 4 or 5 turn)
           7: 'C'     # coil (residues which are not in any of the above conformations).
}


SS_8_TO_3_ENCODED = {0: 0,
                     1: 0,
                     2: 0,
                     3: 1,
                     4: 1,
                     5: 2,
                     6: 2,
                     7: 2}

SS_8_TO_3 = {'G': 'H',
             'H': 'H',
             'I': 'H',
             'B': 'E',
             'E': 'E',
             'S': 'C',
             'T': 'C',
             'C': 'C'}

SS_3_DICT = {0: 'H',
             1: 'E',
             2: 'C'}

TM_DICT = {0: 'I',
           1: 'S',
           2: 'O',
           3: 'M'}
