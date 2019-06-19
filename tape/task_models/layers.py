from typing import Optional
import tensorflow as tf
from tensorflow.keras import Model
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer, CuDNNLSTM

import numpy as np

from rinokeras.layers import WeightNormDense as Dense
from rinokeras.layers import ApplyAttentionMask


class ComputeClassVector(Model):

    def __init__(self):
        super().__init__()
        self.compute_attention = Dense(1, activation='linear')
        self.attention_mask = ApplyAttentionMask()

    def call(self, inputs, mask=None):
        inputs.shape.assert_has_rank(3)
        attention_weight = self.compute_attention(inputs)

        if mask is not None and mask.shape.ndims == 2:
            mask = mask[:, :, None]
        attention_weight = self.attention_mask(attention_weight, mask=mask)
        attention = tf.nn.softmax(attention_weight, 1)

        output = tf.squeeze(tf.matmul(inputs, attention, transpose_a=True), 2)
        return output


class ScorePairwiseSimilarity(Layer):

    def build(self, input_shape):
        input_shape.assert_has_rank(2)
        d_model = input_shape[-1].value
        self.similarity_matrix = self.add_weight(
            'similarity_matrix',
            dtype=tf.float32,
            shape=[d_model, d_model],
            initializer='glorot_uniform')

    def call(self, vector):
        return tf.matmul(
            tf.matmul(vector, self.similarity_matrix),
            vector, transpose_b=True) / np.sqrt(vector.shape.as_list()[-1])


class BidirectionalCudnnLSTM(Model):

    def __init__(self,
                 units,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 return_sequences=False,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.return_sequences = return_sequences

        if not return_sequences:
            raise NotImplementedError("Must set return_sequences=True")

        self.forward_lstm = CuDNNLSTM(
            units, kernel_initializer, recurrent_constraint, bias_initializer,
            unit_forget_bias, kernel_regularizer, recurrent_regularizer,
            bias_regularizer, activity_regularizer, kernel_constraint,
            recurrent_constraint, bias_constraint, return_sequences=True)

        self.reverse_lstm = CuDNNLSTM(
            units, kernel_initializer, recurrent_constraint, bias_initializer,
            unit_forget_bias, kernel_regularizer, recurrent_regularizer,
            bias_regularizer, activity_regularizer, kernel_constraint,
            recurrent_constraint, bias_constraint, return_sequences=True)

    def call(self, inputs, sequence_lengths):
        forward_output = self.forward_lstm(inputs)

        reversed_sequence = tf.reverse_sequence(inputs, sequence_lengths, seq_axis=1)
        reverse_output = self.reverse_lstm(reversed_sequence)
        reverse_output = tf.reverse_sequence(reverse_output, sequence_lengths, seq_axis=1)

        output_sequence = tf.concat((forward_output, reverse_output), -1)

        return output_sequence
