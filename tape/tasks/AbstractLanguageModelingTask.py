from typing import Tuple, List, Dict, Union
import os
from glob import glob
import pickle as pkl

import tensorflow as tf
import rinokeras as rk
from tape.losses import classification_loss_and_accuracy

from .Task import SequenceToSequenceClassificationTask


class AbstractLanguageModelingTask(SequenceToSequenceClassificationTask):

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

        ece = tf.exp(loss)
        probs = tf.nn.softmax(logits)
        logp = tf.nn.log_softmax(logits)
        perplexity = tf.exp(-tf.reduce_sum(probs * logp, -1))
        weights = tf.ones_like(perplexity) * tf.cast(mask, perplexity.dtype)
        perplexity = tf.reduce_sum(perplexity * weights) / (tf.reduce_sum(weights) + 1e-10)

        metrics = {self.key_metric: accuracy, 'ECE': ece, 'Perplexity': perplexity}
        return loss, metrics

    def get_train_files(self, data_folder) -> List[str]:
        train_files = glob(os.path.join(data_folder, 'pfam', '*train_*[0-9].tfrecord'))
        if len(train_files) == 0:
            raise FileNotFoundError("No training TFrecord files found in directory")
        return train_files

    def get_valid_files(self, data_folder) -> List[str]:
        valid_files = glob(os.path.join(data_folder, 'pfam', '*valid_*[0-9].tfrecord'))
        if len(valid_files) == 0:
            raise FileNotFoundError("No validation TFrecord files found in directory")
        return valid_files

    def prepare_dataset(self,  # type: ignore
                        filenames: tf.data.Dataset,
                        buckets: List[int],
                        batch_sizes: List[int],
                        shuffle: bool,
                        is_holdout: bool,
                        holdout_clans: set,
                        holdout_families: set) -> tf.data.Dataset:

        def _check_membership(tensor, array):
            iscontained = tf.py_func(lambda t: t in array, [tensor], tf.bool)
            iscontained.set_shape(())
            return iscontained

        def _filter_fn(example):
            is_holdout_example = \
                _check_membership(example['clan'], holdout_clans) | \
                _check_membership(example['family'], holdout_families)
            return ~ (is_holdout ^ is_holdout_example)

        def _load_records_and_preprocess(fname: tf.Tensor):
            dataset = tf.data.TFRecordDataset(fname)
            dataset = dataset.map(self._deserialization_func)
            # Hold out a prespecified set of families and clans
            dataset = dataset.filter(_filter_fn)
            return dataset

        dataset = filenames.apply(
            tf.data.experimental.parallel_interleave(
                _load_records_and_preprocess,
                sloppy=True,
                cycle_length=128,
                buffer_output_elements=32))

        dataset = dataset.shuffle(1024) if shuffle else dataset.prefetch(1024)
        batch_fun = tf.data.experimental.bucket_by_sequence_length(
            lambda example: example['protein_length'],
            buckets,
            batch_sizes)
        dataset = dataset.apply(batch_fun)
        return dataset

    def get_data(self,
                 boundaries: Tuple[List[int], List[int]],
                 data_folder: str,
                 max_sequence_length: int = 100000,
                 add_cls_token: bool = False,
                 **kwargs) -> Tuple[tf.data.Dataset, tf.data.Dataset]:

        fam_file = os.path.join(data_folder, 'pfam', 'pfam_fams.pkl')
        clan_file = os.path.join(data_folder, 'pfam', 'pfam_clans.pkl')

        _holdout_clans = ['CL0635', 'CL0624', 'CL0355', 'CL0100', 'CL0417', 'CL0630']
        _holdout_families = ['PF18346', 'PF14604', 'PF18697', 'PF03577', 'PF01112', 'PF03417']

        with open(fam_file, 'rb') as f:
            fam_dict: Dict[str, int] = pkl.load(f)

        with open(clan_file, 'rb') as f:
            clan_dict: Dict[str, int] = pkl.load(f)

        holdout_clans = {clan_dict[k] for k in _holdout_clans}
        holdout_families = {fam_dict[k] for k in _holdout_families}

        print('Currently holding out the following families:', *_holdout_families, sep='\n-')
        print('Currently holding out the following clans: ', *_holdout_clans, sep='\n-')

        train_files = self.get_train_files(data_folder)
        valid_files = self.get_valid_files(data_folder)
        train_files = [fname for fname in train_files if fname not in valid_files]

        train_filenames = tf.data.Dataset.from_tensor_slices(tf.constant(train_files))
        valid_filenames = tf.data.Dataset.from_tensor_slices(tf.constant(valid_files))

        buckets, batch_sizes = boundaries
        train_data = self.prepare_dataset(
            train_filenames, buckets, batch_sizes, shuffle=True, is_holdout=False,
            holdout_clans=holdout_clans, holdout_families=holdout_families)
        valid_data = self.prepare_dataset(
            valid_filenames, buckets, batch_sizes, shuffle=False, is_holdout=False,
            holdout_clans=holdout_clans, holdout_families=holdout_families)

        return train_data, valid_data

    def get_test_data(self,
                      boundaries: Tuple[List[int], List[int]],
                      datafile: Union[str, List[str]],
                      **kwargs) -> tf.data.Dataset:

        if isinstance(datafile, str):
            datafile = [datafile]

        if not all(map(os.path.exists, datafile)):
            raise FileNotFoundError(datafile)

        buckets, batch_sizes = boundaries

        filenames = tf.data.Dataset.from_tensor_slices(tf.constant(datafile))

        test_data = self.prepare_dataset(
            filenames, buckets, batch_sizes, shuffle=False, is_holdout=False,
            holdout_clans=set(), holdout_families=set())

        return test_data
