from constants import *
import tensorflow as tf
from collections import defaultdict


def make_kb_vocab(infile):
    file = open(infile)
    lines = file.readlines()
    lines = [line.strip() for line in lines]
    # raw_vocab = defaultdict()
    id_vocab = defaultdict()
    for idx, line in enumerate(lines):
        if idx != 0:
            pairs = line.split('\t')
            name, name_id = pairs[0], pairs[1]
            # raw_vocab[name_id] = name
            id_vocab[name_id] = idx
    return id_vocab


def parse_all(fileins):
    all_lines = []
    for filein in fileins:
        with open(filein) as f:
            lines = f.readlines()
        all_lines.extend(lines)
    all_words = []
    all_ents = []
    all_head_mask = []
    all_e1_mask = []
    all_e2_mask = []
    all_identities = []
    all_labels = []
    pmid = ''
    all_chems = []
    all_dis = []

    chem_vocab = make_kb_vocab(DATA + 'chemical2id.txt')
    dis_vocab = make_kb_vocab(DATA + 'disease2id.txt')

    for line in all_lines:
        l = line.strip().split()
        if len(l) == 1:
            pmid = l[0]
        else:
            pair = l[0]
            label = l[1]
            if label:
                chem, dis = pair.split('_')

                chem_idx = chem_vocab[chem]
                dis_idx = dis_vocab[dis]

                all_chems.append([chem_idx])
                all_dis.append([dis_idx])

                lb = relation.index(label)

                all_labels.append(lb)

                joint_sdp = ' '.join(l[2:])
                token_ids = tokenizer.encode(joint_sdp)
                all_words.append(token_ids)

                e1_ids, e2_ids, e1_ide, e2_ide = None, None, None, None
                for i in range(len(token_ids)):
                    if token_ids[i] == START_E1:
                        e1_ids = i
                    if token_ids[i] == END_E1:
                        e1_ide = i
                    if token_ids[i] == START_E2:
                        e2_ids = i
                    if token_ids[i] == END_E2:
                        e2_ide = i
                pos = [e1_ids, e1_ide, e2_ids, e2_ide]
                all_ents.append(pos)

                all_identities.append((pmid, pair))
            else:
                print(l)

    for t in all_ents:
        m0 = []
        for i in range(MAX_SEN_LEN):
            m0.append(0.0)
        m0[0] = 1.0
        all_head_mask.append(m0)
        m1 = []
        for i in range(MAX_SEN_LEN):
            m1.append(0.0)
        for i in range(t[0], t[1] - 1):
            m1[i] = 1 / (t[1] - 1 - t[0])
        all_e1_mask.append(m1)
        m2 = []
        for i in range(MAX_SEN_LEN):
            m2.append(0.0)
        for i in range(t[2] - 2, t[3] - 3):
            m2[i] = 1 / ((t[3] - 3) - (t[2] - 2))
        all_e2_mask.append(m2)

    return all_words, all_head_mask, all_e1_mask, all_e2_mask, all_identities, all_labels, all_chems, all_dis


def mat_mul(hidden_output, e_mask):
    e_mask = tf.expand_dims(e_mask, 1)
    e_mask = tf.cast(e_mask, tf.float32)
    prod = e_mask @ hidden_output
    prod = tf.squeeze(prod, axis=1)
    return prod
