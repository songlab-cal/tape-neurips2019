from typing import Optional, Tuple, List

import numpy as np
from sacred import Ingredient

import rinokeras as rk
from rinokeras.models.transformer import TransformerInputEmbedding, TransformerEncoder

from .AbstractTapeModel import AbstractTapeModel


transformer_hparams = Ingredient('transformer')


@transformer_hparams.config
def configure_transformer():
    n_layers = 12  # noqa: F841
    n_heads = 8  # noqa: F841
    d_model = 512  # noqa: F841
    d_filter = 4 * d_model  # noqa: F841
    dropout = 0.1  # noqa: F841
    layer_dropout = 0.  # noqa: F841
    kernel_regularizer = None  # noqa: F841


class Transformer(AbstractTapeModel):

    @transformer_hparams.capture
    def __init__(self,
                 n_symbols: int,
                 n_layers: int = 12,
                 n_heads: int = 8,
                 d_model: int = 512,
                 d_filter: int = 2048,
                 dropout: Optional[float] = 0.1,
                 layer_dropout: Optional[float] = None,
                 kernel_regularizer: Optional[str] = None) -> None:
        super().__init__(n_symbols)
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_filter = d_filter
        self.kernel_regularizer = kernel_regularizer
        self.dropout = dropout

        input_embedding = TransformerInputEmbedding(
            d_model, discrete=True, n_symbols=n_symbols, dropout=dropout,
            concat_position_encoding=True, reproject_position_encoding=True)

        self.encoder = TransformerEncoder(
            input_embedding, n_layers, n_heads, d_model, d_filter, dropout, layer_dropout)

        print(self)

    def __str__(self) -> str:
        outstr = []
        outstr.append('Transformer with Parameters:')
        outstr.append(f'\tn_layers: {self.n_layers}')
        outstr.append(f'\tn_heads: {self.n_heads}')
        outstr.append(f'\td_model: {self.d_model}')
        outstr.append(f'\td_filter: {self.d_filter}')
        outstr.append(f'\tdropout: {self.dropout}')
        return '\n'.join(outstr)

    def call(self, inputs):
        """
        Args:
            sequence: tf.Tensor[int32] - Amino acid sequence,
                a padded tensor with shape [batch_size, MAX_PROTEIN_LENGTH]

            protein_length: tf.Tensor[int32] - Length of each protein in the sequence, a tensor with shape [batch_size]

        Output:
            encoder_output: tf.Tensor[float32] - embedding of each amino acid
                a tensor with shape [batch_size, MAX_PROTEIN_LENGTH, d_model]
        """

        sequence = inputs['primary']
        protein_length = inputs['protein_length']

        attention_mask = rk.utils.convert_to_attention_mask(sequence, protein_length)

        encoder_output = self.encoder(sequence, mask=attention_mask)
        inputs['encoder_output'] = encoder_output
        return inputs

    def get_optimal_batch_sizes(self) -> Tuple[List[int], List[int]]:
        bucket_sizes = np.array(
            [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
             1100, 1200, 1300, 1400, 1500, 1600, 1700, 18000, 2000])
        batch_sizes = np.array(
            [4, 3, 2, 1.5, 1, 0.9, 0.9, 0.8, 0.65, 0.6,
             0.5, 0.5, 0.4, 0.3, 0.3, 0.2, 0.2, 0.1, 0, 0])

        batch_sizes = np.asarray(batch_sizes * self._get_gpu_memory(), np.int32)
        batch_sizes[batch_sizes <= 0] = 1

        return bucket_sizes, batch_sizes
