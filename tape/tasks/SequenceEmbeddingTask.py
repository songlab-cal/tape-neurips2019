from typing import Dict, Tuple, List, Callable

import tensorflow as tf

from tape.data_utils.serialize_fasta import deserialize_protein_sequence

from .Task import Task


class SequenceEmbeddingTask(Task):

    def __init__(self,
                 key_metric: str = 'whocares',
                 deserialization_func: Callable[[bytes], Dict[str, tf.Tensor]] = deserialize_protein_sequence,
                 input_name: str = 'encoder_output',
                 output_name: str = 'encoder_output'):
        super().__init__(key_metric, deserialization_func)
        self._input_name = input_name
        self._output_name = output_name

    def loss_function(self,
                      inputs: Dict[str, tf.Tensor],
                      outputs: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        metrics = {self.key_metric: 1.0}
        loss = 0.0  # loss has to be a float

        return loss, metrics

    def build_output_model(self, layers: List[tf.keras.Model]) -> List[tf.keras.Model]:
        return layers
