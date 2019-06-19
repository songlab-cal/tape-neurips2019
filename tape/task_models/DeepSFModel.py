import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (Lambda, Dense, Dropout, Permute,
                                     Flatten, Conv1D, BatchNormalization,
                                     Activation, Concatenate)

import rinokeras as rk
from rinokeras.layers import Stack

"""
Model from the DeepSF paper.
See github for details: https://github.com/multicom-toolbox/DeepSF/blob/master/lib/library.py
Paper: https://www.ncbi.nlm.nih.gov/pubmed/29228193
"""


class DeepSFConv(Stack):

    def __init__(self,
                 filters: int,
                 kernel_size: int) -> None:
        super().__init__()
        self.add(Conv1D(filters, kernel_size, use_bias=True,
                        kernel_initializer='he_normal', activation='linear',
                        padding='same'))
        self.add(BatchNormalization())
        self.add(Activation('relu'))

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = tf.cast(mask, inputs.dtype)
            if mask.shape.ndims == inputs.shape.ndims - 1:
                mask = tf.expand_dims(mask, -1)
            inputs = inputs * mask
        return super().call(inputs, mask=mask)


class DeepSFBlock(Model):

    def __init__(self,
                 add_checkpoint: bool = False,  # used with memory saving gradients
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self._add_checkpoint = add_checkpoint
        self.conv6 = DeepSFConv(10, 6)
        self.conv10 = DeepSFConv(10, 10)

    def call(self, inputs, mask=None):
        output6 = self.conv6(inputs, mask=mask)
        output10 = self.conv10(inputs, mask=mask)
        output = tf.concat([output6, output10], -1)

        if self._add_checkpoint:
            tf.add_to_collection('checkpoints', output)

        return output


class DeepSFModel(Model):

    def __init__(self, n_classes: int, input_name: str = 'encoder_output', output_name: str = 'logits'):
        super().__init__()
        self._input_name = input_name
        self._output_name = output_name

        def max_pool_30(x):
            maxpool, _ = tf.nn.top_k(x, 30)
            return maxpool

        conv6 = Stack([DeepSFConv(10, 6) for _ in range(10)] + [Permute([2, 1]), Lambda(max_pool_30), Flatten()])
        conv10 = Stack([DeepSFConv(10, 10) for _ in range(10)] + [Permute([2, 1]), Lambda(max_pool_30), Flatten()])

        output_model = Stack()

        # Make conv layers
        output_model.add(Concatenate(-1))
        output_model.add(Dense(500, activation='relu', kernel_initializer='he_normal',
                               kernel_constraint=tf.keras.constraints.max_norm(3)))
        output_model.add(Dropout(0.2))
        output_model.add(Dense(n_classes, kernel_initializer='he_normal'))

        self.conv6 = conv6
        self.conv10 = conv10
        self.output_model = output_model

    def call(self, inputs):
        encoder_output = inputs[self._input_name]
        protein_length = inputs['protein_length']
        mask = rk.utils.convert_sequence_length_to_sequence_mask(
            encoder_output, protein_length)
        conv6 = self.conv6(encoder_output, mask=mask)
        conv10 = self.conv10(encoder_output, mask=mask)
        prediction = self.output_model([conv6, conv10])
        inputs[self._output_name] = prediction
        return inputs
