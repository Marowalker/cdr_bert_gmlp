from utils import *
from constants import *
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import pickle


def make_pickle(fileins, fileouts, case='train'):
    if not os.path.exists(PICKLE_DATA):
        os.makedirs(PICKLE_DATA)

    words, heads, e1s, e2s, identities, labels = parse_all(fileins)

    fo1, fo2, fo3, fo4, fo5 = fileouts

    with open(fo1, 'wb') as f:
        pickle.dump(words, f)
        f.close()

    with open(fo2, 'wb') as f:
        pickle.dump(heads, f)
        f.close()

    with open(fo3, 'wb') as f:
        pickle.dump(e1s, f)
        f.close()

    with open(fo4, 'wb') as f:
        pickle.dump(e2s, f)
        f.close()

    if case == 'train':
        with open(fo5, 'wb') as f:
            pickle.dump(labels, f)
            f.close()

        return words, heads, e1s, e2s, labels
    else:
        with open(fo5, 'wb') as f:
            pickle.dump(identities, f)
            f.close()
        return words, heads, e1s, e2s, identities


def load_pickle(fileins, case='train'):
    fi1, fi2, fi3, fi4, fi5 = fileins

    with open(fi1, 'rb') as f:
        words = pickle.load(f)
        f.close()

    with open(fi2, 'rb') as f:
        heads = pickle.load(f)
        f.close()

    with open(fi1, 'rb') as f:
        e1s = pickle.load(f)
        f.close()

    with open(fi1, 'rb') as f:
        e2s = pickle.load(f)
        f.close()

    if case == 'train':
        with open(fi1, 'rb') as f:
            labels = pickle.load(f)
            f.close()
        return words, heads, e1s, e2s, labels

    else:
        with open(fi1, 'rb') as f:
            identities = pickle.load(f)
            f.close()
        return words, heads, e1s, e2s, identities


def get_x(train_x):
    train_x = pad_sequences(train_x, maxlen=MAX_SEN_LEN, padding='post')
    train_x = tf.constant(train_x)
    return train_x


def get_x_mask(train_x_head_mask, train_x_e1_mask, train_x_e2_mask):
    train_x_head_mask = tf.constant(train_x_head_mask)
    train_x_e1_mask = tf.constant(train_x_e1_mask)
    train_x_e2_mask = tf.constant(train_x_e2_mask)

    return train_x_head_mask, train_x_e1_mask, train_x_e2_mask
