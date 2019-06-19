import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Lambda

import rinokeras as rk
from rinokeras.layers import Stack


class BeplerContactPredictor(Model):

    def __init__(self,
                 input_name: str = 'encoder_output',
                 output_name: str = 'contact_prob'):
        super().__init__()
        self._input_name = input_name
        self._output_name = output_name

        def concat_pairs(tensor):
            input_mul = tensor[:, :, None] * tensor[:, None, :]
            input_sub = tf.abs(tensor[:, :, None] - tensor[:, None, :])
            output = tf.concat((input_mul, input_sub), -1)
            return output

        self.get_pairwise_feature_vector = Lambda(concat_pairs)

        self.predict_contact_map = Stack()
        self.predict_contact_map.add(Conv2D(32, 1, use_bias=True, padding='same', activation='relu'))
        self.predict_contact_map.add(Conv2D(1, 7, use_bias=True, padding='same', activation='linear'))

    def call(self, inputs):
        encoder_output = inputs[self._input_name]
        tf.add_to_collection('checkpoints', encoder_output)

        z = self.get_pairwise_feature_vector(encoder_output)
        sequence_mask = rk.utils.convert_sequence_length_to_sequence_mask(
            encoder_output, inputs['protein_length'])
        mask_2d = sequence_mask[:, None, :] & sequence_mask[:, :, None]

        prediction = self.predict_contact_map(z, mask=mask_2d)
        inputs[self._output_name] = prediction
        return inputs
