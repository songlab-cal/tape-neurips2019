import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dropout

import rinokeras as rk
from rinokeras.layers import WeightNormDense as Dense
from rinokeras.layers import Stack, LayerNorm, ApplyAttentionMask


class ComputeClassVector(Model):

    def __init__(self,
                 input_name: str = 'encoder_output',
                 output_name: str = 'cls_vector'):
        super().__init__()
        self._input_name = input_name
        self._output_name = output_name
        self.compute_attention = Stack([LayerNorm(), Dense(1, activation='linear'), Dropout(0.1)])
        self.attention_mask = ApplyAttentionMask()

    def call(self, inputs):
        sequence_mask = rk.utils.convert_sequence_length_to_sequence_mask(
            inputs['primary'], inputs['protein_length'])

        encoder_output = inputs[self._input_name]
        attention_weight = self.compute_attention(encoder_output)

        attention_weight = self.attention_mask(
            attention_weight, mask=sequence_mask[:, :, None])
        attention = tf.nn.softmax(attention_weight, 1)

        cls_vector = tf.squeeze(tf.matmul(encoder_output, attention, transpose_a=True), 2)
        inputs[self._output_name] = cls_vector

        return inputs
