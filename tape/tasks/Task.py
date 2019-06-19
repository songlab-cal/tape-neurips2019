from abc import ABC, abstractmethod
import operator
import os
import re
from typing import Dict, Tuple, List, Callable

import numpy as np
import tensorflow as tf
import rinokeras as rk

from tape.task_models import AminoAcidClassPredictor, GlobalVectorPredictor, ComputeClassVector

from tape.losses import classification_loss_and_accuracy, binary_loss_and_accuracy


class Task(ABC):

    def __init__(self,
                 key_metric: str,
                 deserialization_func: Callable[[bytes], Dict[str, tf.Tensor]]):
        self._key_metric = key_metric
        self._deserialization_func = deserialization_func

    def get_train_files(self, data_folder: str) -> List[str]:
        train_file = os.path.join(data_folder, str(self), '{}_train.tfrecords'.format(self))
        if not os.path.exists(train_file):
            raise FileNotFoundError(train_file)

        return [train_file]

    def get_valid_files(self, data_folder: str) -> List[str]:
        valid_file = os.path.join(data_folder, str(self), '{}_valid.tfrecords'.format(self))
        if not os.path.exists(valid_file):
            raise FileNotFoundError(valid_file)

        return [valid_file]

    def __str__(self) -> str:
        name = self.__class__.__name__[:-4]  # remove the Task
        # Convert from camel to snake
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
        return name

    def prepare_dataset(self,
                        dataset: tf.data.Dataset,
                        buckets: List[int],
                        batch_sizes: List[int],
                        shuffle: bool = False) -> tf.data.Dataset:
        dataset = dataset.map(self._deserialization_func, num_parallel_calls=128)

        buckets_array = np.array(buckets)
        batch_sizes_array = np.array(batch_sizes)

        if np.any(batch_sizes_array == 0) and shuffle:
            iszero = np.where(batch_sizes_array == 0)[0][0]
            filterlen = buckets_array[iszero - 1]
            print("Filtering sequences of length {}".format(filterlen))
            dataset = dataset.filter(lambda example: example['protein_length'] < filterlen)
        else:
            batch_sizes_array[batch_sizes_array <= 0] = 1

        dataset = dataset.shuffle(1024) if shuffle else dataset.prefetch(1024)
        batch_fun = tf.data.experimental.bucket_by_sequence_length(
            operator.itemgetter('protein_length'),
            buckets_array,
            batch_sizes_array)
        dataset = dataset.apply(batch_fun)
        return dataset

    @abstractmethod
    def loss_function(self,
                      inputs: Dict[str, tf.Tensor],
                      outputs: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        raise NotImplementedError

    def get_data(self,
                 boundaries: Tuple[List[int], List[int]],
                 data_folder: str,
                 max_sequence_length: int = 100000,
                 add_cls_token: bool = False,
                 **kwargs) -> Tuple[tf.data.Dataset, tf.data.Dataset]:

        train_files = self.get_train_files(data_folder)
        valid_files = self.get_valid_files(data_folder)

        train_data = tf.data.TFRecordDataset(train_files)
        valid_data = tf.data.TFRecordDataset(valid_files)

        buckets, batch_sizes = boundaries
        train_data = self.prepare_dataset(train_data, buckets, batch_sizes, shuffle=True)
        valid_data = self.prepare_dataset(valid_data, buckets, batch_sizes, shuffle=False)

        return train_data, valid_data

    def get_test_data(self,
                      boundaries: Tuple[List[int], List[int]],
                      datafile: str,
                      **kwargs) -> tf.data.Dataset:

        if not os.path.exists(datafile):
            raise FileNotFoundError(datafile)

        test_data = tf.data.TFRecordDataset(datafile)

        buckets, batch_sizes = boundaries
        test_data = self.prepare_dataset(test_data, buckets, batch_sizes, shuffle=False)

        return test_data

    @abstractmethod
    def build_output_model(self, layers: List[tf.keras.Model]) -> List[tf.keras.Model]:
        raise NotImplementedError

    @property
    def key_metric(self) -> str:
        return self._key_metric


class SequenceToSequenceClassificationTask(Task):

    def __init__(self,
                 key_metric: str,
                 deserialization_func: Callable[[bytes], Dict[str, tf.Tensor]],
                 n_classes: int,
                 label_name: str,
                 input_name: str = 'encoder_output',
                 output_name: str = 'sequence_logits',
                 mask_name: str = 'sequence_mask'):
        super().__init__(key_metric, deserialization_func)
        self._n_classes = n_classes
        self._label_name = label_name
        self._input_name = input_name
        self._output_name = output_name
        self._mask_name = mask_name

    def loss_function(self,
                      inputs: Dict[str, tf.Tensor],
                      outputs: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        labels = inputs[self._label_name]
        logits = outputs[self._output_name]
        if self._mask_name != 'sequence_mask':
            mask = outputs[self._mask_name]
        else:
            mask = rk.utils.convert_sequence_length_to_sequence_mask(
                labels, inputs['protein_length'])
        loss, accuracy = classification_loss_and_accuracy(
            labels, logits, mask)
        metrics = {self.key_metric: accuracy}
        return loss, metrics

    def build_output_model(self, layers: List[tf.keras.Model]) -> List[tf.keras.Model]:
        layers.append(AminoAcidClassPredictor(self._n_classes, self._input_name, self._output_name))
        return layers


class SequenceToFloatTask(Task):

    def __init__(self,
                 key_metric: str,
                 deserialization_func: Callable[[bytes], Dict[str, tf.Tensor]],
                 d_output: int,
                 label: str,
                 input_name: str = 'encoder_output',
                 output_name: str = 'prediction'):
        super().__init__(key_metric, deserialization_func)
        self._d_output = d_output
        self._label = label
        self._input_name = input_name
        self._output_name = output_name

    def loss_function(self,
                      inputs: Dict[str, tf.Tensor],
                      outputs: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        label = inputs[self._label]
        prediction = outputs[self._output_name]

        mse = tf.losses.mean_squared_error(label, prediction)
        mae = tf.losses.absolute_difference(label, prediction)

        metrics = {self.key_metric: mae}

        return mse, metrics

    def build_output_model(self, layers: List[tf.keras.Model]) -> List[tf.keras.Model]:
        layers.append(ComputeClassVector(self._input_name, 'cls_vector'))
        layers.append(GlobalVectorPredictor(self._d_output, 'cls_vector', self._output_name))
        return layers


class SequenceClassificationTask(Task):

    def __init__(self,
                 key_metric: str,
                 deserialization_func: Callable[[bytes], Dict[str, tf.Tensor]],
                 n_classes: int,
                 label: str,
                 input_name: str = 'encoder_output',
                 output_name: str = 'logits'):
        super().__init__(key_metric, deserialization_func)
        self._n_classes = n_classes
        self._label = label
        self._input_name = input_name
        self._output_name = output_name

    def loss_function(self,
                      inputs: Dict[str, tf.Tensor],
                      outputs: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        label = inputs[self._label]
        prediction = outputs[self._output_name]

        loss, accuracy = classification_loss_and_accuracy(label, prediction)

        metrics = {self.key_metric: accuracy}

        return loss, metrics

    def build_output_model(self, layers: List[tf.keras.Model]) -> List[tf.keras.Model]:
        layers.append(ComputeClassVector(self._input_name, 'cls_vector'))
        layers.append(GlobalVectorPredictor(self._n_classes, 'cls_vector', self._output_name))
        return layers


class SequenceBinaryClassificationTask(Task):

    def __init__(self,
                 key_metric: str,
                 deserialization_func: Callable[[bytes], Dict[str, tf.Tensor]],
                 label: str,
                 input_name: str = 'encoder_output',
                 output_name: str = 'prediction'):
        super().__init__(key_metric, deserialization_func)
        self._label = label
        self._input_name = input_name
        self._output_name = output_name

    def loss_function(self,
                      inputs: Dict[str, tf.Tensor],
                      outputs: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        label = inputs[self._label]
        prediction = outputs[self._output_name]
        loss, accuracy = binary_loss_and_accuracy(label, prediction)

        metrics = {self.key_metric: accuracy}

        return loss, metrics

    def build_output_model(self, layers: List[tf.keras.Model]) -> List[tf.keras.Model]:
        layers.append(ComputeClassVector(self._input_name, 'cls_vector'))
        layers.append(GlobalVectorPredictor(1, 'cls_vector', self._output_name))
        return layers
