import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv1D, Dense

import rinokeras as rk
from rinokeras.layers import PaddedConv, Stack

from .layers import BidirectionalCudnnLSTM


class NetsurfModel(Model):

    def __init__(self, input_name: str = 'encoder_output'):
        super().__init__()
        self._input_name = input_name

        self.convs = Stack([
            PaddedConv(1, 32, 129),
            PaddedConv(1, 32, 257)])
        self.bilstm = Stack([
            BidirectionalCudnnLSTM(1024, return_sequences=True),
            BidirectionalCudnnLSTM(1024, return_sequences=True)])

        # Need to predict phi, psi, rsa, disorder, ss3, ss8
        num_outputs = 0
        num_outputs += 1  # phi
        num_outputs += 2  # psi
        num_outputs += 1  # rsa
        num_outputs += 1  # disorder
        num_outputs += 1  # interface
        num_outputs += 3  # ss3
        num_outputs += 8  # ss8

        self.predict_outputs = Dense(num_outputs, activation=None)

    def call(self, inputs):
        encoder_output = inputs[self._input_name]
        protein_length = inputs['protein_length']
        mask = rk.utils.convert_sequence_length_to_sequence_mask(
            encoder_output, protein_length)
        conv_out = self.convs(encoder_output, mask=mask)
        features = tf.concat((encoder_output, conv_out), -1)
        tf.add_to_collection('checkpoints', features)

        lstm_out = self.bilstm(features, sequence_lengths=protein_length)
        tf.add_to_collection('checkpoints', lstm_out)

        outputs = self.predict_outputs(lstm_out)

        phi = outputs[:, :, 0]
        psi = outputs[:, :, 1]
        rsa = outputs[:, :, 2]
        disorder = outputs[:, :, 3]
        interface = outputs[:, :, 4]
        ss3 = outputs[:, :, 5:8]
        ss8 = outputs[:, :, 8:16]

        inputs.update({
            'phi_pred': phi,
            'psi_pred': psi,
            'rsa_pred': rsa,
            'disorder_pred': disorder,
            'interface_pred': interface,
            'ss3_pred': ss3,
            'ss8_pred': ss8})

        return inputs
