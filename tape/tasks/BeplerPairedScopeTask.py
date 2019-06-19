from typing import Tuple, List, Dict

import numpy as np
import tensorflow as tf
import rinokeras as rk
from rinokeras.layers import Stack

from tape.data_utils import deserialize_paired_sequence
from tape.task_models import SoftSymmetricAlignment, OrdinalRegression
from .Task import Task


class BeplerPairedScopeTask(Task):

    def __init__(self):
        super().__init__(key_metric='PairedACC', deserialization_func=deserialize_paired_sequence)

    def loss_function(self,
                      inputs: Dict[str, tf.Tensor],
                      outputs: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        # Reshape logits for multiclass setup
        true_labels = tf.range(5)[None] <= inputs['label']
        true_labels = tf.cast(true_labels, tf.int32)

        probs = tf.nn.sigmoid(outputs['ordinal_logits'])
        cumprobs = rk.utils.gather_from_last(probs, inputs['label'])
        nextprobs = rk.utils.gather_from_last(
            tf.pad(probs, [[0, 0], [0, 1]]), inputs['label'] + 1)

        accuracy = tf.reduce_mean(cumprobs * (1 - nextprobs))

        # Now get loss and accuracy
        loss = tf.losses.sigmoid_cross_entropy(
            multi_class_labels=true_labels,
            logits=outputs['ordinal_logits'])
        metrics = {self.key_metric: accuracy}
        return loss, metrics

    def prepare_dataset(self,
                        dataset: tf.data.Dataset,
                        buckets: List[int],
                        batch_sizes: List[int],
                        shuffle: bool = False) -> tf.data.Dataset:
        dataset = dataset.map(self._deserialization_func, 128)
        dataset = dataset.shuffle(1024) if shuffle else dataset.prefetch(1024)
        batch_fun = tf.data.experimental.bucket_by_sequence_length(
            lambda example: tf.maximum(example['first']['protein_length'],
                                       example['second']['protein_length']),
            buckets,
            batch_sizes)
        dataset = dataset.apply(batch_fun)
        return dataset

    def get_data(self,
                 boundaries: Tuple[List[int], List[int]],
                 data_folder: str,
                 max_sequence_length: int,
                 add_cls_token: bool,
                 **kwargs) -> Tuple[tf.data.Dataset, tf.data.Dataset]:

        bounds, batch_sizes = boundaries
        batch_sizes = np.asarray(np.sqrt(batch_sizes), np.int32) // 8
        batch_sizes[batch_sizes == 0] = 1

        return super().get_data(
            (bounds, batch_sizes), data_folder, max_sequence_length, add_cls_token, **kwargs)

    def build_output_model(self, layers: List[tf.keras.Model]) -> List[tf.keras.Model]:
        ssa = SoftSymmetricAlignment(Stack(layers))
        return [ssa, OrdinalRegression(5)]
