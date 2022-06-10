from constants import *
import tensorflow as tf


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
    for line in all_lines:
        l = line.strip().split()
        if len(l) == 1:
            pmid = l[0]
        else:
            pair = l[0]
            label = l[1]
            if label:
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

    return all_words, all_head_mask, all_e1_mask, all_e2_mask, all_identities, all_labels


def mat_mul(hidden_output, e_mask):
    e_mask = tf.expand_dims(e_mask, 1)
    e_mask = tf.cast(e_mask, tf.float32)
    prod = e_mask @ hidden_output
    prod = tf.squeeze(prod, axis=1)
    return prod
