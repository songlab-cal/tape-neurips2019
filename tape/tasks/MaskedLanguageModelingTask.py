from typing import List

import tensorflow as tf
from sacred import Ingredient

from tape.data_utils import PFAM_VOCAB, deserialize_pfam_sequence
from tape.task_models import RandomSequenceMask, AminoAcidClassPredictor
from .AbstractLanguageModelingTask import AbstractLanguageModelingTask


mask_params = Ingredient('mask')


@mask_params.config
def mask_config():
    percentage = 0.15  # noqa: F841
    style = 'random'  # noqa: F841


class MaskedLanguageModelingTask(AbstractLanguageModelingTask):

    @mask_params.capture
    def __init__(self,
                 percentage: float = 0.15,
                 style: str = 'random'):
        n_symbols = len(PFAM_VOCAB)
        mask_token = PFAM_VOCAB['<MASK>']

        super().__init__(
            key_metric='BERTAcc',
            deserialization_func=deserialize_pfam_sequence,
            n_classes=n_symbols,
            label_name='original_sequence',
            input_name='encoder_output',
            output_name='bert_logits',
            mask_name='bert_mask')
        self._mask_token = mask_token
        self._percentage = percentage
        self._style = style

    def build_output_model(self, layers: List[tf.keras.Model]) -> List[tf.keras.Model]:
        layers.insert(0, RandomSequenceMask(self._n_classes, self._mask_token, self._percentage, self._style))
        layers.append(AminoAcidClassPredictor(self._n_classes, self._input_name, self._output_name, use_conv=True))
        return layers
