from typing import Optional

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
import numpy as np

import rinokeras as rk
from rinokeras.layers import WeightNormDense as Dense
from rinokeras.layers import LayerNorm, Stack


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
                 mask_token: int,
                 n_symbols: Optional[int] = None,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not 0 <= percentage < 1:
            raise ValueError("Masking percentage must be in [0, 1).\
                Received {}".format(percentage))
        self.percentage = percentage
        self.mask_token = mask_token
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

        bert_mask = self._generate_bert_mask(inputs)

        if mask is not None:
            bert_mask &= mask

        masked_inputs = inputs * tf.cast(~bert_mask, inputs.dtype)

        token_bert_mask = K.random_uniform(K.shape(bert_mask)) < 0.8
        random_bert_mask = (K.random_uniform(
            K.shape(bert_mask)) < 0.1) & ~token_bert_mask
        true_bert_mask = ~token_bert_mask & ~random_bert_mask

        token_bert_mask = tf.cast(token_bert_mask & bert_mask, inputs.dtype)
        random_bert_mask = tf.cast(random_bert_mask & bert_mask, inputs.dtype)
        true_bert_mask = tf.cast(true_bert_mask & bert_mask, inputs.dtype)

        masked_inputs += self.mask_token * token_bert_mask  # type: ignore

        masked_inputs += K.random_uniform(
            K.shape(bert_mask), 0, self.n_symbols, dtype=inputs.dtype) * random_bert_mask

        masked_inputs += inputs * true_bert_mask

        return masked_inputs, bert_mask


class ContiguousReplaceMask(Layer):
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
                 mask_token: int,
                 n_symbols: Optional[int] = None,
                 avg_seq_len: int = 3,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not 0 <= percentage < 1:
            raise ValueError("Masking percentage must be in [0, 1).\
                Received {}".format(percentage))
        self.percentage = percentage
        self.mask_token = mask_token
        self.avg_seq_len = avg_seq_len
        self.n_symbols = n_symbols

    def _generate_bert_mask(self, inputs):

        def _numpy_generate_contiguous_mask(array):
            mask = np.random.random(array.shape) < (1 / self.avg_seq_len)
            mask = np.cumsum(mask, 1)
            seqvals = np.max(mask)
            mask_prob = self.percentage * array.shape[1] / seqvals  # increase probability because fewer sequences
            vals_to_mask = np.arange(seqvals)[np.random.random((seqvals,)) < mask_prob]
            indices_to_mask = np.isin(mask, vals_to_mask)
            mask[indices_to_mask] = 1
            mask[~indices_to_mask] = 0

            return np.asarray(mask, np.bool)

        bert_mask = tf.py_func(_numpy_generate_contiguous_mask, [inputs], tf.bool)
        bert_mask.set_shape(inputs.shape)
        return bert_mask


class RandomSequenceMask(Model):

    def __init__(self,
                 n_symbols: int,
                 mask_token: int,
                 mask_percentage: float = 0.15,
                 mask_type: str = 'random'):
        super().__init__()

        if mask_type == 'random':
            self.bert_mask = RandomReplaceMask(mask_percentage, mask_token, n_symbols)
        elif mask_type == 'contiguous':
            self.bert_mask = ContiguousReplaceMask(mask_percentage, mask_token, n_symbols)
        else:
            raise ValueError("Unrecognized mask_type: {}".format(mask_type))

    def call(self, inputs):
        """
        Args:
            sequence: tf.Tensor[int32] - Amino acid sequence,
                a padded tensor with shape [batch_size, MAX_PROTEIN_LENGTH]

            protein_length: tf.Tensor[int32] - Length of each protein in the sequence, a tensor with shape [batch_size]

        Output:
            amino_acid_probs: tf.Tensor[float32] - Probability of each type of amino acid,
                a tensor with shape [batch_size, MAX_PROTEIN_LENGTH, n_symbols]
        """

        sequence = inputs['primary']
        protein_length = inputs['protein_length']

        sequence_mask = rk.utils.convert_sequence_length_to_sequence_mask(
            sequence, protein_length)
        masked_sequence, bert_mask = self.bert_mask(sequence, sequence_mask)

        inputs['original_sequence'] = sequence
        inputs['primary'] = masked_sequence
        inputs['bert_mask'] = bert_mask

        return inputs
