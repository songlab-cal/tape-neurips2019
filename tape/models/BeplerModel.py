from typing import Tuple, List, Optional

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Dropout, Dense, Activation
from tensorflow.keras.layers import LSTM, Bidirectional, Layer
from sacred import Ingredient
import numpy as np

import rinokeras as rk
from rinokeras.layers import Stack

from tape.data_utils.vocabs import PFAM_VOCAB, UNIPROT_BEPLER
from .AbstractTapeModel import AbstractTapeModel

bepler_hparams = Ingredient('bepler')


@bepler_hparams.config
def configure_bepler():
    dropout = 0.1  # noqa: F841
    use_pfam_alphabet: bool = True  # noqa: F841


class RandomReplaceMask(Layer):
    """ Copied from rinokeras because we're going to potentially have
    different  replace masks.

    Replaces some percentage of the input with a mask token. Used for
    implementing  style models. This is actually slightly more complex - it
    does one of three things

    Based on https://arxiv.org/abs/1810.04805.

    Args:
        percentage (float): Percentage of input tokens to mask
        mask_token (int): Token to replace masked input with
    """

    def __init__(self,
                 percentage: float,
                 n_symbols: Optional[int] = None,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not 0 <= percentage < 1:
            raise ValueError("Masking percentage must be in [0, 1).\
                Received {}".format(percentage))
        self.percentage = percentage
        self.n_symbols = n_symbols

    def _generate_bert_mask(self, inputs):
        mask_shape = K.shape(inputs)
        bert_mask = K.random_uniform(mask_shape) < self.percentage
        return bert_mask

    def call(self,
             inputs: tf.Tensor,
             mask: Optional[tf.Tensor] = None):
        """
        Args:
            inputs (tf.Tensor[ndims=2, int]): Tensor of values to mask
            mask (Optional[tf.Tensor[bool]]): Locations in the inputs to that are valid
                                                     (i.e. not padding, start tokens, etc.)
        Returns:
            masked_inputs (tf.Tensor[ndims=2, int]): Tensor of masked values
            bert_mask: Locations in the input that were masked
        """

        random_mask = self._generate_bert_mask(inputs)

        if mask is not None:
            random_mask &= mask

        masked_inputs = inputs * tf.cast(~random_mask, inputs.dtype)

        random_mask = tf.cast(random_mask, inputs.dtype)

        masked_inputs += K.random_uniform(
            K.shape(random_mask), 0, self.n_symbols, dtype=inputs.dtype) * random_mask

        return masked_inputs


class BiLM(Model):

    @bepler_hparams.capture
    def __init__(self, n_symbols: int, dropout: float = 0, use_pfam_alphabet: bool = True):
        super().__init__()

        self._use_pfam_alphabet = use_pfam_alphabet

        if use_pfam_alphabet:
            self.embed = Embedding(n_symbols, n_symbols)
        else:
            n_symbols = 21
            self.embed = Embedding(n_symbols + 1, n_symbols)

        self.dropout = Dropout(dropout)
        self.rnn = Stack([
            LSTM(1024, return_sequences=True, use_bias=True,
                 implementation=2, recurrent_activation='sigmoid'),
            LSTM(1024, return_sequences=True, use_bias=True,
                 implementation=2, recurrent_activation='sigmoid')])

        self.compute_logits = Dense(n_symbols, use_bias=True, activation='linear')

    def transform(self, z_fwd, z_rvs, mask_fwd, mask_rvs, sequence_lengths):
        h_fwd = []
        h = z_fwd

        for layer in self.rnn.layers:
            h = layer(h, mask=mask_fwd)
            h = self.dropout(h)
            h_fwd.append(h)

        h_rvs = []
        h = z_rvs
        for layer in self.rnn.layers:
            h = layer(h, mask=mask_rvs)
            h = self.dropout(h)
            h_rvs.append(
                tf.reverse_sequence(h, sequence_lengths - 1, seq_axis=1))

        return h_fwd, h_rvs

    def embed_and_split(self, x, sequence_lengths, pad=False):
        if pad:
            # Add one to each sequence element
            if not self._use_pfam_alphabet:
                x = x + 1
                mask = rk.utils.convert_sequence_length_to_sequence_mask(x, sequence_lengths)
                x = x * tf.cast(mask, x.dtype)

            x = tf.pad(x, [[0, 0], [1, 1]])  # pad x
            sequence_lengths += 2

        mask = rk.utils.convert_sequence_length_to_sequence_mask(x, sequence_lengths)

        z = self.embed(x)
        z_fwd = z[:, :-1]
        mask_fwd = mask[:, :-1]

        z_rvs = tf.reverse_sequence(z, sequence_lengths, seq_axis=1)[:, :-1]
        mask_rvs = tf.reverse_sequence(mask, sequence_lengths, seq_axis=1)[:, :-1]

        return z_fwd, z_rvs, mask_fwd, mask_rvs, sequence_lengths

    def call(self, inputs, encode=False):
        inputs, sequence_lengths = inputs
        z_fwd, z_rvs, mask_fwd, mask_rvs, sequence_lengths = self.embed_and_split(
            inputs, sequence_lengths, pad=encode)
        h_fwd_list, h_rvs_list = self.transform(
            z_fwd, z_rvs, mask_fwd, mask_rvs, sequence_lengths)

        h_fwd = h_fwd_list[-1]
        h_rvs = h_rvs_list[-1]

        lm_outputs = tf.concat((h_fwd[:, 1:], h_rvs[:, :-1]), -1)
        logp_fwd = self.compute_logits(h_fwd)
        logp_rvs = self.compute_logits(h_rvs)

        # prepend forward logp with zero
        # postpend reverse logp with zero
        logp_fwd = tf.pad(logp_fwd, [[0, 0], [1, 0], [0, 0]])
        logp_rvs = tf.pad(logp_rvs, [[0, 0], [0, 1], [0, 0]])

        logp = tf.nn.log_softmax(logp_fwd + logp_rvs)

        concat = []
        for h_fwd, h_rvs in zip(h_fwd_list, h_rvs_list):
            h_fwd = h_fwd[:, :-1]
            h_rvs = h_rvs[:, 1:]

            concat.extend([h_fwd, h_rvs])

        h = tf.concat(concat, -1)

        return {'logp': logp, 'h': h, 'lm_outputs': lm_outputs}


class LMEmbed(Model):

    @bepler_hparams.capture
    def __init__(self, n_symbols: int, dropout: float = 0, use_pfam_alphabet: bool = True):
        super().__init__()

        if not use_pfam_alphabet:
            n_symbols = 21

        self.embed = Embedding(n_symbols, 512)
        self.lm = BiLM(n_symbols, dropout)
        self.proj = Dense(512, use_bias=True, activation='linear')
        self.transform = Activation('relu')

    def call(self, inputs):
        inputs, sequence_lengths = inputs
        h_lm = self.lm((inputs, sequence_lengths), encode=True)
        lm_outputs = h_lm['lm_outputs']
        h_lm = h_lm['h']

        h = self.embed(inputs)
        h_lm = self.proj(h_lm)
        h = self.transform(h + h_lm)
        return h, lm_outputs


class BeplerModel(AbstractTapeModel):

    @bepler_hparams.capture
    def __init__(self,
                 n_symbols: int,
                 dropout: float = 0,
                 use_pfam_alphabet: bool = True):
        if not use_pfam_alphabet:
            n_symbols = 21

        super().__init__(n_symbols)
        self._use_pfam_alphabet = use_pfam_alphabet

        self.embed = LMEmbed(n_symbols, dropout)
        self.dropout = Dropout(dropout)
        lstm = Stack([
            Bidirectional(
                LSTM(512, return_sequences=True, use_bias=True,
                     recurrent_activation='sigmoid', implementation=2))
            for _ in range(3)])
        self.rnn = lstm
        self.proj = Dense(100, use_bias=True, activation='linear')
        self.random_replace = RandomReplaceMask(0.05, n_symbols)

    def convert_sequence_vocab(self, sequence):
        PFAM_TO_BEPLER_ENCODED = {encoding: UNIPROT_BEPLER.get(aa, 20) for aa, encoding in PFAM_VOCAB.items()}
        PFAM_TO_BEPLER_ENCODED[PFAM_VOCAB['<PAD>']] = 0

        def to_uniprot_bepler(seq):
            new_seq = np.zeros_like(seq)

            for pfam_encoding, uniprot_encoding in PFAM_TO_BEPLER_ENCODED.items():
                new_seq[seq == pfam_encoding] = uniprot_encoding

            return new_seq

        new_sequence = tf.py_func(to_uniprot_bepler, [sequence], sequence.dtype)
        new_sequence.set_shape(sequence.shape)

        return new_sequence

    def call(self, inputs):
        sequence = inputs['primary']
        protein_length = inputs['protein_length']

        if not self._use_pfam_alphabet:
            sequence = self.convert_sequence_vocab(sequence)
        sequence = K.in_train_phase(self.random_replace(sequence), sequence)

        mask = rk.utils.convert_sequence_length_to_sequence_mask(sequence, protein_length)
        embed, lm_outputs = self.embed((sequence, protein_length))
        tf.add_to_collection('checkpoints', embed)
        rnn_out = self.rnn(embed, mask=mask)
        tf.add_to_collection('checkpoints', rnn_out)
        rnn_out = self.dropout(rnn_out)
        proj = self.proj(rnn_out)
        tf.add_to_collection('checkpoints', proj)

        inputs['encoder_output'] = proj
        inputs['lm_outputs'] = lm_outputs

        return inputs

    def get_optimal_batch_sizes(self) -> Tuple[List[int], List[int]]:
        bucket_sizes = np.array([100, 200, 300, 400, 600, 900, 1000, 1200, 1300, 2000, 3000])
        batch_sizes = np.array([5, 5, 4, 3, 2, 1, 0.75, 0.5, 0.5, 0.25, 0, 0])

        batch_sizes = np.asarray(batch_sizes * self._get_gpu_memory(), np.int32)
        return bucket_sizes, batch_sizes
