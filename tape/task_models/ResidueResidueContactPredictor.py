import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Lambda

import rinokeras as rk
from rinokeras.layers import Stack, LayerNorm, ResidualBlock, PaddedConv
from rinokeras.layers import WeightNormDense as Dense


class ResidueResidueContactPredictor(Model):

    def __init__(self,
                 input_name: str = 'encoder_output',
                 output_name: str = 'sequence_logits'):
        super().__init__()
        self._input_name = input_name
        self._output_name = output_name

        self.get_pairwise_feature_vector = Stack()
        self.get_pairwise_feature_vector.add(Dense(64, activation='linear'))

        def concat_pairs(tensor):
            seqlen = tf.shape(tensor)[1]
            input_left = tf.tile(tensor[:, :, None], (1, 1, seqlen, 1))
            input_right = tf.tile(tensor[:, None, :], (1, seqlen, 1, 1))
            output = tf.concat((input_left, input_right), -1)
            return output

        self.get_pairwise_feature_vector.add(Lambda(concat_pairs))

        self.predict_contact_map = Stack()
        self.predict_contact_map.add(
            PaddedConv(2, 64, 1, dropout=0.1))
        for layer in range(30):
            self.predict_contact_map.add(
                ResidualBlock(2, 64, 3, dropout=0.1, add_checkpoint=layer % 5 == 0))
        self.predict_contact_map.add(Dense(1, activation='linear'))

    def call(self, inputs):
        encoder_output = inputs[self._input_name]
        tf.add_to_collection('checkpoints', encoder_output)

        z = self.get_pairwise_feature_vector(encoder_output)
        self.pairwise_z = z
        sequence_mask = rk.utils.convert_sequence_length_to_sequence_mask(
            encoder_output, inputs['protein_length'])
        mask_2d = sequence_mask[:, None, :] & sequence_mask[:, :, None]

        prediction = self.predict_contact_map(z, mask=mask_2d)
        prediction = (prediction + tf.transpose(prediction, (0, 2, 1, 3))) / 2  # symmetrize
        inputs[self._output_name] = prediction
        return inputs
