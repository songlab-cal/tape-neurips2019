from typing import Union, List
import tensorflow as tf
from tensorflow.keras.layers import Layer


class BidirectionalOutputShift(Layer):

    def __init__(self,
                 input_name: str,
                 output_name: str):
        super().__init__()
        self._input_name = input_name
        self._output_name = output_name

    def call(self, inputs):
        # Take the off-by-one next sentence into account
        # For the forward output, ignore the last output, for reverse ignore first
        forward_output, reverse_output = tf.split(inputs[self._input_name], axis=-1, num_or_size_splits=2)
        forward_output = tf.pad(forward_output[:, :-1], [[0, 0], [1, 0], [0, 0]])
        reverse_output = tf.pad(reverse_output[:, 1:], [[0, 0], [0, 1], [0, 0]])

        lm_output = tf.concat((forward_output, reverse_output), -1)

        inputs[self._output_name] = lm_output

        return inputs
