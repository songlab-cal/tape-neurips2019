from typing import Tuple, List
from abc import ABC, abstractmethod
import subprocess
from tensorflow.keras import Model


class AbstractTapeModel(Model, ABC):

    def __init__(self, n_symbols: int):
        super().__init__()
        self._n_symbols = n_symbols
        self._input_key = 'primary'

    def _get_gpu_memory(self) -> int:
        """Returns number of GB in the gpu. Allows very stupid auto-scaling to different
        memory sizes."""
        nvidia_smi = subprocess.check_output('nvidia-smi')
        memsize = list(filter(lambda word: 'MiB' in word, nvidia_smi.decode().split()))[1]
        memsize = int(memsize[:-3]) // 1000  # number of gigabytes on gpu
        return memsize

    @abstractmethod
    def get_optimal_batch_sizes(self) -> Tuple[List[int], List[int]]:
        """
        Returns a list of sequence lengths and a list of batch sizes
        corresponding to the optimal batch size for that sequence length.

        Can use self._get_gpu_memory() to access GPU memory on current machine.
        See docs for tf.data.experimental.bucket_by_sequence_length for details
        - the first return should correspond to the `bucket_boundaries` and the
        second return should correspond to the `bucket_batch_sizes` argument.
        """

        return NotImplemented

    @property
    def input_key(self) -> str:
        """Whatever we're calling the amino acid sequence after deserializing"""
        return self._input_key
