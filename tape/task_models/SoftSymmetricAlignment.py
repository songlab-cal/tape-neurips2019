import tensorflow as tf

import tensorflow.keras.backend as K
from tensorflow.keras import Model
import rinokeras as rk


class SoftSymmetricAlignment(Model):

    def __init__(self, embedding_model: Model) -> None:
        super().__init__()
        self.embedding_model = embedding_model

    def call(self, inputs):
        if 'first' not in inputs or 'second' not in inputs:
            return self.embedding_model(inputs)

        first = inputs['first']
        second = inputs['second']

        output_x = self.embedding_model(first)
        output_y = self.embedding_model(second)

        ssa = self.soft_symmetric_alignment(
            output_x['encoder_output'], output_y['encoder_output'],
            first['protein_length'], second['protein_length'])[:, None]

        inputs['ssa'] = ssa
        return inputs

    def soft_symmetric_alignment(self, x, y, x_length, y_length):
        # Taken from Tristan:
        # https://github.com/tbepler/protein-sequence-embedding-iclr2019/blob/master/src/models/comparison.py

        # x and y of shape: [batch_size, seq_len, embed_dim]
        x_mask = rk.utils.convert_sequence_length_to_sequence_mask(x, x_length)
        y_mask = rk.utils.convert_sequence_length_to_sequence_mask(y, y_length)

        attention_mask = K.cast(x_mask[:, :, None] & y_mask[:, None, :], K.floatx())

        tf.add_to_collection('checkpoints', x)
        tf.add_to_collection('checkpoints', y)

        # z_diff is a tensor of ||z_i - z_j||_1 of shape: [batch_size, seq_len, seq_len]
        z_diff = tf.norm(x[:, :, None, :] - y[:, None, :, :], ord=1, axis=-1)

        weights = - z_diff - 1e9 * (1 - attention_mask)

        alpha = tf.nn.softmax(weights, axis=1)
        beta = tf.nn.softmax(weights, axis=2)
        a = attention_mask * (alpha + beta - alpha * beta)
        ssa = - tf.reduce_sum(a * z_diff, axis=[1, 2]) / tf.reduce_sum(a, axis=[1, 2])
        tf.add_to_collection('checkpoints', ssa)

        return ssa


# Including this sanity test here
# import tensorflow as tf
# import numpy as np

# tf.enable_eager_execution()

# x = tf.cast(tf.expand_dims(np.array([[1, 2, 3], [2, 3, 4]]), 0), tf.float32)
# y = tf.cast(tf.expand_dims(np.array([[5, 6, 7], [10, 11, 12], [14, 15, 16]]), 0), tf.float32)
# tf.norm(tf.expand_dims(x, axis=seq_length_axis) - y, ord=1, axis=-1)
# soft_symmetric_alignment(x, y)


# In [21]: tf.norm(tf.expand_dims(x, axis=seq_length_axis) - y, ord=1, axis=-1)
# Out[21]:
# <tf.Tensor: id=51, shape=(1, 2, 2), dtype=float32, numpy=
# array([[[12., 27.],
#         [ 9., 24.]]], dtype=float32)>

# In [23]: soft_symmetric_alignment(x, y)
# Out[23]: <tf.Tensor: id=102, shape=(1,), dtype=float32, numpy=array([-15.047427], dtype=float32)>
