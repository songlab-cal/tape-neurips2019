from typing import Optional, Tuple, List

from tensorflow.keras.layers import Embedding, Lambda
import numpy as np
from sacred import Ingredient

import rinokeras as rk
from rinokeras.layers import Stack, ResidualBlock, PaddedConv, PositionEmbedding

from .AbstractTapeModel import AbstractTapeModel


resnet_hparams = Ingredient('resnet')


@resnet_hparams.config
def configure_resnet():
    n_layers = 35  # noqa: F841
    filters = 256  # noqa: F841
    kernel_size = 9  # noqa: F841
    layer_norm = False  # noqa: F841
    activation = 'relu'  # noqa: F841
    dilation_rate = 2  # noqa: F841
    dropout = 0.1  # noqa: F841


class Resnet(AbstractTapeModel):

    @resnet_hparams.capture
    def __init__(self,
                 n_symbols: int,
                 n_layers: int = 35,
                 filters: int = 256,
                 kernel_size: int = 9,
                 layer_norm: bool = True,
                 activation: str = 'elu',
                 dilation_rate: int = 2,
                 dropout: Optional[float] = 0.1) -> None:
        super().__init__(n_symbols)
        self.n_symbols = n_symbols
        self.n_layers = n_layers
        self.filters = filters
        self.kernel_size = kernel_size
        self.layer_norm = layer_norm
        self.activation = activation
        self.dilation_rate = dilation_rate
        self.dropout = dropout

        print(self)

        input_embedding = Stack()
        input_embedding.add(Embedding(n_symbols, 128))
        input_embedding.add(Lambda(lambda x: x * np.sqrt(filters)))
        input_embedding.add(PositionEmbedding())

        encoder = Stack()
        encoder.add(input_embedding)
        encoder.add(PaddedConv(1, filters, kernel_size, 1, activation, dropout))
        encoder.add(ResidualBlock(1, filters, kernel_size, activation=activation,
                                  dilation_rate=1, dropout=dropout))
        for layer in range(n_layers - 1):
            encoder.add(ResidualBlock(1, filters, kernel_size, activation=activation,
                                      dilation_rate=dilation_rate, dropout=dropout,
                                      add_checkpoint=layer % 5 == 0))

        self.encoder = encoder

    def __str__(self) -> str:
        outstr = []
        outstr.append('Resnet with Parameters:')
        outstr.append(f'\tn_layers: {self.n_layers}')
        outstr.append(f'\tfilters: {self.filters}')
        outstr.append(f'\tkernel_size: {self.kernel_size}')
        outstr.append(f'\tactivation: {self.activation}')
        outstr.append(f'\tdilation_rate: {self.dilation_rate}')
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
                a tensor with shape [batch_size, MAX_PROTEIN_LENGTH, filters]
        """

        sequence = inputs['primary']
        protein_length = inputs['protein_length']

        sequence_mask = rk.utils.convert_sequence_length_to_sequence_mask(sequence, protein_length)
        encoder_output = self.encoder(sequence, mask=sequence_mask)

        inputs['encoder_output'] = encoder_output
        return inputs

    def get_optimal_batch_sizes(self) -> Tuple[List[int], List[int]]:
        bucket_sizes = np.array([100, 200, 300, 400, 600, 900, 1000, 1300, 2000, 3000])
        batch_sizes = np.array([4, 4, 4, 4, 3, 3, 3, 2, 1, 0.5, 0])

        batch_sizes = np.asarray(batch_sizes * self._get_gpu_memory(), np.int32)
        batch_sizes[batch_sizes <= 0] = 1
        return bucket_sizes, batch_sizes
