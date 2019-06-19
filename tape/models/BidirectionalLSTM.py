from typing import Optional, Tuple, List

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import CuDNNLSTM as LSTM
from tensorflow.keras.layers import Embedding, Dropout
from sacred import Ingredient

from rinokeras.layers import Stack
from .AbstractTapeModel import AbstractTapeModel


lstm_hparams = Ingredient('lstm')


@lstm_hparams.config
def configure_lstm():
    n_units = 1024  # noqa: F841
    n_layers = 3  # noqa: F841
    dropout = 0.1  # noqa: F841


class BidirectionalLSTM(AbstractTapeModel):

    @lstm_hparams.capture
    def __init__(self,
                 n_symbols: int,
                 n_units: int = 1024,
                 n_layers: int = 3,
                 dropout: Optional[float] = 0.1) -> None:
        super().__init__(n_symbols)

        if dropout is None:
            dropout = 0

        self.embedding = Embedding(n_symbols, 128)

        self.forward_lstm = Stack([
            LSTM(n_units,
                 return_sequences=True) for _ in range(n_layers)],
            name='forward_lstm')

        self.reverse_lstm = Stack([
            LSTM(n_units,
                 return_sequences=True) for _ in range(n_layers)],
            name='reverse_lstm')

        self.dropout = Dropout(dropout)

    def call(self, inputs):
        sequence = inputs['primary']
        protein_length = inputs['protein_length']

        sequence = self.embedding(sequence)
        tf.add_to_collection('checkpoints', sequence)

        forward_output = self.forward_lstm(sequence)
        tf.add_to_collection('checkpoints', forward_output)

        reversed_sequence = tf.reverse_sequence(sequence, protein_length, seq_axis=1)
        reverse_output = self.reverse_lstm(reversed_sequence)
        reverse_output = tf.reverse_sequence(reverse_output, protein_length, seq_axis=1)
        tf.add_to_collection('checkpoints', reverse_output)

        encoder_output = tf.concat((forward_output, reverse_output), -1)

        encoder_output = self.dropout(encoder_output)

        inputs['encoder_output'] = encoder_output
        return inputs

    def get_optimal_batch_sizes(self) -> Tuple[List[int], List[int]]:
        bucket_sizes = np.array([100, 200, 300, 400, 600, 900, 1000, 1200, 1300, 2000, 3000])
        batch_sizes = np.array([5, 5, 4, 3, 2, 1, 0.75, 0.5, 0.5, 0.25, 0, 0])

        batch_sizes = np.asarray(batch_sizes * self._get_gpu_memory(), np.int32)
        batch_sizes[batch_sizes <= 0] = 1
        return bucket_sizes, batch_sizes
