from typing import List

import tensorflow as tf

from tape.data_utils import PFAM_VOCAB, deserialize_pfam_sequence
from tape.task_models import AminoAcidClassPredictor
from .AbstractLanguageModelingTask import AbstractLanguageModelingTask


class UnidirectionalLanguageModelingTask(AbstractLanguageModelingTask):

    def __init__(self):
        n_symbols = len(PFAM_VOCAB)
        super().__init__(
            key_metric='LMACC',
            deserialization_func=deserialize_pfam_sequence,
            n_classes=n_symbols,
            label_name='primary',
            input_name='lm_output',
            output_name='lm_logits')

    def build_output_model(self, layers: List[tf.keras.Model]) -> List[tf.keras.Model]:
        layers.append(AminoAcidClassPredictor(self._n_classes, self._input_name, self._output_name, use_conv=False))
        return layers
