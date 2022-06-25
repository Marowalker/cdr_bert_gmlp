from utils import mat_mul
from preprocessing import get_x, get_x_mask
import tensorflow as tf
from constants import *
import numpy as np
from simple_gmlp import gMLPLayer
import os
import keras.backend as K


def f1_macro(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    # tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


class BertgMLPModel:
    def __init__(self, base_encoder, depth, chem_emb, dis_emb):
        if not os.path.exists(TRAINED_MODELS):
            os.makedirs(TRAINED_MODELS)

        self.encoder = base_encoder
        self.depth = depth
        self.chem_emb = chem_emb
        self.dis_emb = dis_emb

        self.input_ids = None
        self.head_mask = None
        self.e1_mask = None
        self.e2_mask = None
        self.chem_ids = None
        self.dis_ids = None

    def _add_inputs(self):
        self.input_ids = tf.keras.layers.Input(shape=(MAX_SEN_LEN,), dtype='int32')
        self.head_mask = tf.keras.layers.Input(shape=(MAX_SEN_LEN,))
        self.e1_mask = tf.keras.layers.Input(shape=(MAX_SEN_LEN,))
        self.e2_mask = tf.keras.layers.Input(shape=(MAX_SEN_LEN,))
        self.chem_ids = tf.keras.layers.Input(shape=(1,), dtype='int32')
        self.dis_ids = tf.keras.layers.Input(shape=(1,), dtype='int32')

    def _bert_layer(self):
        self.bertoutput = self.encoder(self.input_ids)
        emb = self.bertoutput[0]

        chem = tf.keras.layers.embeddings.Embedding(self.chem_emb.shape[0] + 1, EMB_SIZE, weights=[self.chem_emb],
                                                    trainable=False)(self.chem_emb)
        dis = tf.keras.layers.embeddings.Embedding(self.dis_emb.shape[0] + 1, EMB_SIZE, weights=[self.dis_emb],
                                                   trainable=False)(self.dis_emb)

        x = gMLPLayer(dropout_rate=0.05)(emb)
        for _ in range(self.depth - 1):
            x = gMLPLayer(dropout_rate=0.05)(x)

        chem_x = gMLPLayer(dropout_rate=0.05)(chem)
        for _ in range(self.depth - 1):
            chem_x = gMLPLayer(dropout_rate=0.05)(chem_x)

        dis_x = gMLPLayer(dropout_rate=0.05)(dis)
        for _ in range(self.depth - 1):
            dis_x = gMLPLayer(dropout_rate=0.05)(chem_x)

        cls = mat_mul(x, self.head_mask)
        cls = tf.keras.layers.Dropout(DROPOUT)(cls)
        cls = tf.keras.layers.Dense(EMB_SIZE, activation='tanh')(cls)

        e1 = mat_mul(chem_x, self.e1_mask)
        e1 = tf.keras.layers.Dropout(DROPOUT)(e1)
        e1 = tf.keras.layers.Dense(EMB_SIZE, activation='tanh')(e1)

        e2 = mat_mul(dis_x, self.e2_mask)
        e2 = tf.keras.layers.Dropout(DROPOUT)(e2)
        e2 = tf.keras.layers.Dense(EMB_SIZE, activation='tanh')(e2)
        # e2 = tf.keras.layers.Dense(EMB_SIZE, activation='relu')(e2)

        com = tf.keras.layers.concatenate([cls, e1, e2])

        out = tf.keras.layers.Dropout(DROPOUT)(com)
        out = tf.keras.layers.Dense(len(relation), activation='softmax')(out)
        return out

    @staticmethod
    def f1_score(y_true, y_pred):
        return f1_macro(y_true, y_pred)

    def _add_train_ops(self):
        self.model = tf.keras.Model(inputs=[self.input_ids, self.head_mask, self.e1_mask, self.e2_mask, self.chem_ids, self.dis_ids],
                                    outputs=self._bert_layer())
        # model = tf.keras.Model(inputs=[input_ids, head_mask], outputs=out)
        # model = tf.keras.Model(inputs=[input_ids, e1_mask, e2_mask], outputs=out)
        self.optimizer = tf.keras.optimizers.Adam(lr=LEARNING_RATE)

        self.model.compile(optimizer=self.optimizer, loss='binary_crossentropy', metrics=['accuracy', self.f1_score])
        print(self.model.summary())

    def _train(self, train_x, train_x_head_mask, train_x_e1_mask, train_x_e2_mask, train_y, train_chem, train_dis):

        train_x = get_x(train_x)
        train_x_head_mask, train_x_e1_mask, train_x_e2_mask = get_x_mask(train_x_head_mask, train_x_e1_mask,
                                                                         train_x_e2_mask)
        train_y = tf.keras.utils.to_categorical(train_y)

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_f1_score', patience=3)

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=TRAINED_MODELS,
            save_weights_only=True,
            monitor='val_f1_score',
            mode='auto',
            save_best_only=True)

        self.model.fit([train_x, train_x_head_mask, train_x_e1_mask, train_x_e2_mask, train_chem, train_dis], train_y, validation_split=0.2,
                       batch_size=BATCH_SIZE, epochs=NUM_EPOCH, callbacks=[early_stopping, model_checkpoint_callback])

        self.model.save_weights(TRAINED_MODELS)

    def plot_model(self):
        tf.keras.utils.plot_model(self.model, to_file='model.png', show_shapes=False, show_layer_names=True,
                                  rankdir='TB',
                                  expand_nested=False, dpi=300)

    def build(self, train_x, train_x_head_mask, train_x_e1_mask, train_x_e2_mask, train_y, train_chem, train_dis):
        with tf.device('/device:GPU:0'):
            self._add_inputs()
            self._add_train_ops()
            if REMAKE == 1:
                self._train(train_x, train_x_head_mask, train_x_e1_mask, train_x_e2_mask, train_y, train_chem, train_dis)
                self.plot_model()

    def predict(self, test_x, test_x_head_mask, test_x_e1_mask, test_x_e2_mask):
        self.model.load_weights(TRAINED_MODELS)
        test_x = get_x(test_x)
        test_x_head_mask, test_x_e1_mask, test_x_e2_mask = get_x_mask(test_x_head_mask, test_x_e1_mask,
                                                                      test_x_e2_mask)
        pred = self.model.predict([test_x, test_x_head_mask, test_x_e1_mask, test_x_e2_mask])

        y_pred = []
        for logit in pred:
            y_pred.append(np.argmax(logit))
        return y_pred
