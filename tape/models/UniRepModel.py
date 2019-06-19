from typing import Tuple, List

import tensorflow as tf
from tensorflow.keras import Model, activations, regularizers, initializers, constraints
from tensorflow.keras.layers import RNN, Embedding, Dense, Dropout

from rinokeras.layers import WeightNormDense
import numpy as np
from sacred import Ingredient

from tape.data_utils.vocabs import PFAM_VOCAB, UNIREP_VOCAB
from .AbstractTapeModel import AbstractTapeModel

unirep_hparams = Ingredient('unirep')


@unirep_hparams.config
def configure_unirep():
    n_units = 1900  # noqa: F841
    dropout = 0.1  # noqa: F841
    use_pfam_alphabet: bool = True  # noqa: F841


class mLSTMCell(Model):

    def __init__(self,
                 units,
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.state_size = (self.units, self.units)
        self.output_size = self.units
        self._dropout_mask = None
        self._recurrent_dropout_mask = None

        self.kernel = WeightNormDense(
            self.units * 4,
            use_bias=False,
            activation='linear',
            name='kernel',
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            kernel_constraint=self.kernel_constraint)

        self.recurrent_kernel = WeightNormDense(
            self.units * 4,
            use_bias=True,
            activation='linear',
            name='recurrent_kernel',
            kernel_initializer=self.recurrent_initializer,
            kernel_regularizer=self.recurrent_regularizer,
            kernel_constraint=self.recurrent_constraint)

        self.m_kernel = WeightNormDense(
            self.units,
            use_bias=False,
            activation='linear',
            name='m_kernel',
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            kernel_constraint=self.kernel_constraint)

        self.m_recurrent_kernel = WeightNormDense(
            self.units,
            use_bias=False,
            activation='linear',
            name='m_recurrent_kernel',
            kernel_initializer=self.recurrent_initializer,
            kernel_regularizer=self.recurrent_regularizer,
            kernel_constraint=self.recurrent_constraint)

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, state):
        h_tm1, c_tm1 = state

        m = self.m_kernel(inputs) * self.m_recurrent_kernel(h_tm1)
        z = self.kernel(inputs) + self.recurrent_kernel(m)

        i, f, o, u = tf.split(z, num_or_size_splits=4, axis=1)

        i = self.recurrent_activation(i)
        f = self.recurrent_activation(f)
        o = self.recurrent_activation(o)
        c = f * c_tm1 + i * self.activation(u)

        h = o * self.activation(c)

        return h, [h, c]


class mLSTM(RNN):

    def __init__(self,
                 units,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        cell = mLSTMCell(units, **kwargs)
        super().__init__(
            cell, return_sequences, return_state,
            go_backwards, stateful, unroll)


class UniRepModel(AbstractTapeModel):

    @unirep_hparams.capture
    def __init__(self,
                 n_symbols: int,
                 n_units: int = 1900,
                 dropout: float = 0.1,
                 use_pfam_alphabet: bool = True):

        super().__init__(n_symbols)
        self._use_pfam_alphabet = use_pfam_alphabet

        if use_pfam_alphabet:
            self.embed = Embedding(n_symbols, 10)
        else:  # using the UniRep alphabet
            self.embed = Embedding(26, 10)

        self.mlstm = mLSTM(n_units, return_sequences=True)

        # Include the compute logits layer in case anyone ever wants to use this model
        # more directly. We do not use this layer and take the output embedding directly
        self.compute_logits = Dense(25, use_bias=True, activation='linear')
        self.dropout = Dropout(dropout)

    def convert_sequence_vocab(self, sequence, sequence_lengths):
        PFAM_TO_UNIREP_ENCODED = {encoding: UNIREP_VOCAB.get(aa, 23) for aa, encoding in PFAM_VOCAB.items()}

        def to_uniprot_unirep(seq, seqlens):
            new_seq = np.zeros_like(seq)

            for pfam_encoding, unirep_encoding in PFAM_TO_UNIREP_ENCODED.items():
                new_seq[seq == pfam_encoding] = unirep_encoding

            # add start/stop
            new_seq = np.pad(new_seq, [[0, 0], [1, 1]], mode='constant')
            new_seq[:, 0] = UNIREP_VOCAB['<START>']
            new_seq[np.arange(new_seq.shape[0]), seqlens + 1] = UNIREP_VOCAB['<STOP>']

            return new_seq

        new_sequence = tf.py_func(to_uniprot_unirep, [sequence, sequence_lengths], sequence.dtype)
        new_sequence.set_shape([sequence.shape[0], sequence.shape[1] + 2])

        return new_sequence

    def call(self, inputs):
        sequence = inputs['primary']
        protein_length = inputs['protein_length']

        if self._use_pfam_alphabet:
            sequence = tf.pad(sequence, [[0, 0], [1, 0]], constant_values=PFAM_VOCAB['<CLS>'])
        else:
            sequence = self.convert_sequence_vocab(sequence, protein_length)

        embed = self.embed(sequence)
        lstm_out = self.mlstm(embed)
        lstm_out = self.dropout(lstm_out)

        logits = self.compute_logits(lstm_out)  # noqa: F841

        if self._use_pfam_alphabet:
            inputs['encoder_output'] = lstm_out[:, 1:]
            inputs['lm_output'] = lstm_out[:, :-1]
        else:
            inputs['encoder_output'] = lstm_out[:, 1:-1]

        return inputs

    def get_optimal_batch_sizes(self) -> Tuple[List[int], List[int]]:
        bucket_sizes = np.array([100, 200, 300, 400, 600, 900, 1000, 1200, 1300, 2000, 3000])
        batch_sizes = np.array([5, 5, 4, 3, 2, 1, 0.75, 0.5, 0.5, 0.25, 0, 0])

        batch_sizes = np.asarray(batch_sizes * self._get_gpu_memory(), np.int32)
        batch_sizes[batch_sizes <= 0] = 1

        return bucket_sizes, batch_sizes
