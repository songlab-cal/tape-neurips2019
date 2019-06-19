from typing import Dict, Tuple, List

import tensorflow as tf
import numpy as np
import os

import rinokeras as rk
from tape.data_utils import deserialize_secondary_structure
from tape.losses import classification_loss_and_accuracy
from tape.task_models import NetsurfModel
from .Task import Task


class NetsurfTask(Task):

    def __init__(self):
        super().__init__(
            key_metric='SS3ACC',
            deserialization_func=deserialize_secondary_structure)

        self._input_name = 'encoder_output'

    
    def get_train_files(self, data_folder: str) -> List[str]: 
        train_file = os.path.join(data_folder, 'secondary_structure', 'secondary_structure_train.tfrecords')
        if not os.path.exists(train_file):
            raise FileNotFoundError(train_file)

        return [train_file]

    def get_valid_files(self, data_folder: str) -> List[str]:
        valid_file = os.path.join(data_folder, 'secondary_structure', 'secondary_structure_valid.tfrecords')
        if not os.path.exists(valid_file):
            raise FileNotFoundError(valid_file)

        return [valid_file]
    
    def compute_angle_loss(self, phi, psi, phi_pred, psi_pred, sequence_mask):
        valid_mask = tf.cast(tf.not_equal(phi, 0) | tf.not_equal(psi, 0), tf.float32) * sequence_mask

        phi = phi / 360 * np.pi
        psi = psi / 360 * np.pi

        cos_phi_loss = tf.losses.mean_squared_error(tf.cos(phi), tf.cos(phi_pred), valid_mask)
        sin_phi_loss = tf.losses.mean_squared_error(tf.sin(phi), tf.sin(phi_pred), valid_mask)
        cos_psi_loss = tf.losses.mean_squared_error(tf.cos(psi), tf.cos(psi_pred), valid_mask)
        sin_psi_loss = tf.losses.mean_squared_error(tf.sin(psi), tf.sin(psi_pred), valid_mask)

        return cos_phi_loss + sin_phi_loss + cos_psi_loss + sin_psi_loss

    def loss_function(self,
                      inputs: Dict[str, tf.Tensor],
                      outputs: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        sequence_mask = rk.utils.convert_sequence_length_to_sequence_mask(
            inputs['primary'], inputs['protein_length'])
        valid_mask = tf.cast(inputs['valid_mask'], tf.float32)

        sequence_mask = tf.cast(sequence_mask, tf.float32) * valid_mask

        angle_loss = self.compute_angle_loss(
            inputs['phi'], inputs['psi'], outputs['phi_pred'], outputs['psi_pred'], valid_mask)

        rsa_loss = tf.losses.mean_squared_error(inputs['rsa'], outputs['rsa_pred'], sequence_mask)
        disorder_loss = tf.losses.sigmoid_cross_entropy(inputs['disorder'], outputs['disorder_pred'], sequence_mask)
        interface_loss = tf.losses.sigmoid_cross_entropy(inputs['interface'], outputs['interface_pred'], sequence_mask)
        ss3_loss, ss3_acc = classification_loss_and_accuracy(inputs['ss3'], outputs['ss3_pred'], sequence_mask)
        ss8_loss, ss8_acc = classification_loss_and_accuracy(inputs['ss8'], outputs['ss8_pred'], sequence_mask)

        loss = angle_loss + rsa_loss + disorder_loss + interface_loss + ss3_loss + ss8_loss
        metrics = {'SS3ACC': ss3_acc, 'SS8ACC': ss8_acc}

        return loss, metrics

    def build_output_model(self, layers: List[tf.keras.Model]) -> List[tf.keras.Model]:
        layers.append(NetsurfModel(self._input_name))
        return layers
