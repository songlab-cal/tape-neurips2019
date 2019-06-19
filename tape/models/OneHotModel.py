from typing import Tuple, List
import numpy as np
import tensorflow.keras.backend as K

from .AbstractTapeModel import AbstractTapeModel


class OneHotModel(AbstractTapeModel):

    def __init__(self, n_symbols: int):
        super().__init__(n_symbols)

    def call(self, inputs):
        encoder_output = K.one_hot(inputs['primary'], self._n_symbols)
        inputs['encoder_output'] = encoder_output
        return inputs

    def get_optimal_batch_sizes(self) -> Tuple[List[int], List[int]]:
        bucket_sizes = np.array([100, 200, 300, 400, 600, 900, 1000, 1200, 1300, 2000, 3000])
        batch_sizes = np.array([5, 5, 5, 5, 5, 4, 4, 3, 3, 2, 1, 0])

        batch_sizes = np.asarray(batch_sizes * self._get_gpu_memory(), np.int32)
        batch_sizes[batch_sizes <= 0] = 1
        return bucket_sizes, batch_sizes
