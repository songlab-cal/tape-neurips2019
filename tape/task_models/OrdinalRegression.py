import tensorflow as tf

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer


class OrdinalRegression(Layer):

    def __init__(self, n_classes):
        super().__init__()
        self._n_classes = n_classes

    def build(self, input_shape):
        self.ordinal_weights = self.add_weight(
            shape=[1, self._n_classes],
            dtype=K.floatx(),
            name='ord_loss_weights',
            initializer='glorot_uniform')
        self.ordinal_biases = self.add_weight(shape=[self._n_classes],
                                              dtype=K.floatx(),
                                              name='ord_loss_bias',
                                              initializer='zeros')

    def call(self, inputs):
        if 'ssa' not in inputs:
            return inputs

        ordinal_logits = tf.matmul(inputs['ssa'], tf.exp(self.ordinal_weights)) + self.ordinal_biases
        tf.add_to_collection('checkpoints', ordinal_logits)
        inputs['ordinal_logits'] = ordinal_logits
        return inputs
