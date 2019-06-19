from typing import Dict, Tuple, List

import rinokeras as rk
import tensorflow as tf
import numpy as np
from rinokeras.utils import get_shape

from .Task import Task
from tape.data_utils import deserialize_contact_map_sequence
from tape.task_models import BeplerContactPredictor


class BeplerContactMapTask(Task):

    def __init__(self,
                 min_residue_distance: int = 1):
        super().__init__(
            key_metric='ContactACC',
            deserialization_func=deserialize_contact_map_sequence)
        self._label_name = 'contact_map'
        self._input_name = 'encoder_output'
        self._output_name = 'contact_prob'
        self._min_residue_distance = min_residue_distance

    def get_mask(self, inputs):
        """Mask, -1s, things longer than the sequence length, and nearby contacts"""

        valid_mask = inputs['valid_mask']
        # Mask sequence length padding - don't have to do this if we return valid mask
        # sequence_mask = rk.utils.convert_sequence_length_to_sequence_mask(
            # inputs['primary'], inputs['protein_length'])
        # sequence_mask = sequence_mask[:, :, None] & sequence_mask[:, None, :]

        # Mask nearby contacts
        sequence_length = rk.utils.get_shape(inputs['primary'], 1)
        indices = tf.range(sequence_length)
        distance = tf.abs(indices[None, :] - indices[:, None])
        nearby_contact_mask = tf.greater_equal(distance, self._min_residue_distance)

        mask = valid_mask & nearby_contact_mask[None]

        return mask

    def loss_function(self,
                      inputs: Dict[str, tf.Tensor],
                      outputs: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        mask = self.get_mask(inputs)

        label = inputs[self._label_name]
        pred = tf.squeeze(outputs[self._output_name], 3)

        batch_size, seqlen = get_shape(label, (0, 1))

        # with tf.control_dependencies([tf.print(tf.shape(mask), tf.shape(inputs['valid_mask']), batch_size, seqlen)]):
        mask = tf.reshape(mask, (batch_size, seqlen * seqlen))
        label = tf.reshape(label, (batch_size, seqlen * seqlen))
        pred = tf.reshape(pred, (batch_size, seqlen * seqlen))

        mask = tf.cast(mask, pred.dtype)

        loss = tf.losses.sigmoid_cross_entropy(label, pred, mask)

        max_prot_len = tf.reduce_max(inputs['protein_length'])
        top_l_over_5, indices = tf.nn.top_k(pred - (1000 * (1 - mask)), max_prot_len // 5)

        ii, _ = tf.meshgrid(tf.range(batch_size), tf.range(max_prot_len // 5), indexing='ij')
        l5_labels = tf.gather_nd(label, tf.stack([ii, indices], -1))
        l5_mask = tf.gather_nd(mask, tf.stack([ii, indices], -1))

        l5_labels = tf.cast(l5_labels, l5_mask.dtype)
        accuracy = tf.reduce_sum(l5_labels * l5_mask) / (tf.reduce_sum(l5_mask) + 1e-8)

        metrics = {self.key_metric: accuracy}
        return loss, metrics

    def get_data(self,
                 boundaries: Tuple[List[int], List[int]],
                 data_folder: str,
                 max_sequence_length: int = 100000,
                 add_cls_token: bool = False,
                 **kwargs) -> Tuple[tf.data.Dataset, tf.data.Dataset]:

        bounds, batch_sizes = boundaries
        batch_sizes = np.asarray(np.sqrt(batch_sizes) / 4, np.int32)
        batch_sizes[batch_sizes <= 0] = 1
        batch_sizes[:-1][bounds > 1000] = 0
        batch_sizes[-1] = 0
        return super().get_data(
            (bounds, batch_sizes), data_folder, max_sequence_length, add_cls_token)

    def build_output_model(self, layers: List[tf.keras.Model]) -> List[tf.keras.Model]:
        layers.append(BeplerContactPredictor(self._input_name, self._output_name))
        return layers
