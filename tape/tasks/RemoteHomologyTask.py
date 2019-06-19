from typing import List, Dict, Tuple
import tensorflow as tf

from tape.losses import classification_loss_and_top1_top5_top10_accuracies
from tape.data_utils import deserialize_remote_homology_sequence
from tape.task_models import DeepSFModel, ComputeClassVector, GlobalVectorPredictor
from .Task import SequenceClassificationTask


class RemoteHomologyTask(SequenceClassificationTask):
    # Fold-level classification
    def __init__(self):
        n_classes = 1195
        super().__init__(
            key_metric='ACC',
            deserialization_func=deserialize_remote_homology_sequence,
            n_classes=n_classes,
            label='fold_label',
            input_name='encoder_output',
            output_name='logits')

    def loss_function(self,
                      inputs: Dict[str, tf.Tensor],
                      outputs: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        label = inputs[self._label]
        prediction = outputs[self._output_name]

        loss, top_1_accuracy, top_5_accuracy, top_10_accuracy = classification_loss_and_top1_top5_top10_accuracies(
            label, prediction)

        metrics = {self.key_metric: top_1_accuracy, 'TOP5ACC': top_5_accuracy, 'TOP10ACC': top_10_accuracy}

        return loss, metrics

    def build_output_model(self, layers: List[tf.keras.Model]) -> List[tf.keras.Model]:
        layers.append(ComputeClassVector(self._input_name, 'cls_vector'))
        layers.append(GlobalVectorPredictor(self._n_classes, 'cls_vector', self._output_name))
        # layers.append(DeepSFModel(self._n_classes, self._input_name, self._output_name))
        return layers
