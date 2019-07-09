from typing import List, Union, Sequence, Tuple, Optional
from glob import glob
from collections import defaultdict
import atexit
import os
import shutil
import pickle as pkl

import tensorflow as tf
from tensorflow.contrib.distribute import MirroredStrategy

from tape.tasks import TaskBuilder, Task, AbstractLanguageModelingTask
from tape.models import ModelBuilder
from tape.experiments import ProteinExperiment, training_params
import rinokeras as rk

import re
from datetime import datetime

from sacred import Experiment, Ingredient
from sacred.observers import FileStorageObserver
import numpy as np
from table_logger import TableLogger


gpu = Ingredient('gpu')
proteins = Experiment('Unsupervised Protein',
                      ingredients=[gpu, training_params] + ModelBuilder.hparams + TaskBuilder.params)

folder_name = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
logdir = os.environ.get('PROTEIN_LOGDIR', 'results')
if not os.path.isdir('results'):
    os.mkdir('results')
proteins.observers.append(FileStorageObserver.create(os.path.join('results', folder_name)))


def filter_text(text):
    pattern = re.compile(r"Epoch\s+\d+:")
    text = '\n'.join(filter(lambda line: not pattern.match(line), text.split('\n')))
    return text


proteins.captured_out_filter = filter_text


@gpu.config
def gpu_config():
    """Configure the gpu"""
    device = 0  # noqa: F841
    allow_growth = False  # noqa: F841


@proteins.config
def config():
    tasks = []  # noqa: F841
    model = ''  # noqa: F841
    num_epochs = 100  # noqa: F841
    load_from = None  # noqa: F841
    load_task_from = None  # noqa: F841
    patience = 10  # noqa: F841
    freeze_embedding_weights = False  # noqa: F841
    data_folder = './data'  # noqa: F841
    max_sequence_length = 10000  # noqa: F841
    add_cls_token = False  # noqa: F841
    debug = False  # noqa: F841
    save_outputs = False  # noqa: F841
    steps_per_epoch = 10000  # noqa: F841
    datafile = ''  # noqa: F841

    assert len(tasks) > 0
    assert model != ''


@gpu.capture
def setup_tensorflow(device: Union[str, int, Sequence[int], Sequence[str]], allow_growth: bool):
    """Setup tensorflow session according to gpu configuration.

    Args:
        device (Union[str, int, Sequence[int], Sequence[str]]): GPU or list of GPUs to run on
        allow_growth (bool): Whether to capture all memory on gpu or grow as necessary

    Returns:
        sess (tf.Session): Tensorflow Session object as the default session
    """
    if isinstance(device, int):
        device = str(device)
    elif isinstance(device, list):
        device = ', '.join([str(d) for d in device])
    elif not isinstance(device, str):
        raise ValueError("Unrecognized device type. Expected int, str, or list. "
                         "Received {}.".format(type(device)))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # disable tensorflow info logging
    tf.logging.set_verbosity(tf.logging.WARN)

    from tensorflow.python.platform import tf_logging
    try:
        # tensorflow == 1.13
        tf_logging.get_logger().propagate = False
    except AttributeError:
        # tensorflow <= 1.12
        tf_logging._get_logger().propagate = False

    gpu_options = tf.GPUOptions(allow_growth=allow_growth)
    conf = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    sess = tf.get_default_session()
    if sess is None:
        sess = tf.Session(config=conf)
        sess.__enter__()  # type: ignore

    np.set_printoptions(suppress=True)

    return sess


@proteins.capture
def get_data(task_list: List[Task],
             boundaries: Tuple[List[int], List[int]],
             data_folder: str,
             max_sequence_length: int,
             add_cls_token: bool) -> \
        Tuple[tf.data.Dataset, tf.data.Dataset]:

    datasets = [task.get_data(boundaries, data_folder, max_sequence_length, add_cls_token) for task in task_list]
    train, valid = list(zip(*datasets))
    if len(train) > 1:
        train = tf.data.Dataset.zip(train)
        valid = tf.data.Dataset.zip(valid)
    else:
        train = train[0]
        valid = valid[0]

    return train, valid


class MetricEvaluator:

    def __init__(self, key_metric: str, maximize_or_minimize: Optional[str] = None):
        self._key_metric = key_metric
        if maximize_or_minimize is None:
            maximize_or_minimize = 'max' if 'acc' in key_metric.lower() else 'min'
        assert maximize_or_minimize in ['maximize', 'minimize', 'max', 'min']
        self._best_value = float('inf') if maximize_or_minimize in ['minimize', 'min'] else float('-inf')
        self._maximize_or_minimize = max if maximize_or_minimize in ['max', 'maximize'] else min
        self._n_epochs_no_improvement = 0
        self._epoch = 0

    def _maybe_initialize_logger(self, metrics):
        if not hasattr(self, '_logger'):
            columns = ["Epoch"]
            self._names = [name for name in metrics]
            columns += ["T {}".format(name) for name in metrics]
            columns += ["V {}".format(name) for name in metrics]
            columns += ["Best {}".format(self._key_metric)]
            columns += ["Time"]
            self._logger = TableLogger(
                columns=columns,
                float_format='{:.3f}'.format,
                default_colwidth=10)

    def _is_better(self, test_metrics):
        return self._maximize_or_minimize(
            self._best_value, test_metrics[self._key_metric]) != self._best_value

    def check_and_log_metric(self, train_metrics, test_metrics):
        self._maybe_initialize_logger(train_metrics)
        if self._is_better(test_metrics):
            self._best_value = self._maximize_or_minimize(
                self._best_value, test_metrics[self._key_metric])
            self._n_epochs_no_improvement = 0
        else:
            self._n_epochs_no_improvement += 1

        to_log = [self._epoch]
        to_log += [train_metrics[name] for name in self._names]
        to_log += [test_metrics[name] for name in self._names]
        to_log += [self._best_value]
        to_log += [str(round(train_metrics.runtime + test_metrics.runtime)) + 's']
        self._logger(*tuple(to_log))
        self._epoch += 1

    @property
    def was_improvement(self) -> bool:
        return self._n_epochs_no_improvement == 0

    @property
    def n_epochs_no_improvement(self) -> int:
        return self._n_epochs_no_improvement


@proteins.capture
def rename_directory(outdir: str, model, tasks):
    if isinstance(tasks, str):
        tasks = [tasks]
    savedir, basedir = outdir.rsplit('/', 1)
    new_outdir = os.path.join(savedir, '_'.join(tasks + [model, basedir]))
    os.rename(outdir, new_outdir)


@proteins.capture
def cleanup_folders(outdir: str, model, tasks, debug):
    if debug or not glob(os.path.join(outdir, '*.h5')):
        shutil.rmtree(outdir)
    else:
        rename_directory(outdir, model, tasks)


def consolidate_data(outfile, include_hidden: bool = False):

    with open(outfile, 'rb') as f:
        outputs = pkl.load(f)

    data = defaultdict(list)  # type: ignore

    for output in outputs:
        output = output[0]
        length = output['protein_length']
        for key, protein_batch in output.items():
            for protein_length, protein_data in zip(length, protein_batch):
                if np.isscalar(protein_data):
                    data[key].append(protein_data)
                elif protein_data.ndim == 1 and protein_data.dtype in [np.float32, np.float64]:
                    data[key].append(protein_data)
                else:
                    data[key].append(protein_data[:protein_length])

    data = dict(data)

    if not include_hidden:
        del data['encoder_output']

    with open(outfile, 'wb') as f:
        pkl.dump(data, f)


@proteins.command
def eval(_run, _config, tasks: Union[str, List[str]], model: str):
    assert _config['load_task_from'] is not None
    outdir = _run.observers[0].basedir
    atexit.register(cleanup_folders, outdir, debug=True)

    sess = setup_tensorflow()

    if isinstance(tasks, str):
        tasks = [tasks]

    embedding_model = ModelBuilder.build_model(model)
    task_list = TaskBuilder.build_tasks(tasks)
    task_model = TaskBuilder.build_task_model(
        embedding_model, task_list, _config['freeze_embedding_weights'])

    experiment = ProteinExperiment(
        task_model, task_list)

    if not _config['datafile']:
        _, valid_data = get_data(task_list, embedding_model.get_optimal_batch_sizes())
    else:
        datafile = _config['datafile'] if ',' not in _config['datafile'] else _config['datafile'].split(',')
        valid_data = task_list[0].get_test_data(embedding_model.get_optimal_batch_sizes(), datafile)

    test_graph = rk.train.TestGraph.from_experiment(experiment, valid_data)

    sess.run(tf.global_variables_initializer())

    print('Model Parameters: {}'.format(embedding_model.count_params()))
    print('Loading task weights from {}'.format(_config['load_task_from']))

    rk.utils.load_distributed(
        experiment.distribution_strategy, task_model, _config['load_task_from'])

    task_dir = os.path.dirname(_config['load_task_from'])
    outfile = os.path.join(task_dir, 'outputs.pkl')
    print('Saving outputs to {}'.format(outfile))
    test_metrics = test_graph.run_epoch(save_outputs=outfile)
    print(test_metrics.get_average())
    consolidate_data(outfile, include_hidden=True)


@proteins.command
def embed(_run, _config, tasks: Union[str, List[str]], model: str):
    sess = setup_tensorflow()

    if isinstance(tasks, str):
        tasks = [tasks]

    embedding_model = ModelBuilder.build_model(model)
    task_list = TaskBuilder.build_tasks(tasks)
    task_model = TaskBuilder.build_task_model(
        embedding_model, task_list, _config['freeze_embedding_weights'])

    experiment = ProteinExperiment(
        task_model, task_list)

    datafile = _config['datafile'] if ',' not in _config['datafile'] else _config['datafile'].split(',')
    valid_data = task_list[0].get_test_data(embedding_model.get_optimal_batch_sizes(), datafile)

    sess.run(tf.global_variables_initializer())

    test_graph = rk.train.TestGraph.from_experiment(experiment, valid_data)

    print('Model Parameters: {}'.format(embedding_model.count_params()))

    if _config['load_from'] is not None:
        print('Loading weights from {}'.format(_config['load_from']))
        rk.utils.load_distributed(experiment.distribution_strategy, embedding_model, _config['load_from'])

    outfile = 'outputs.pkl'

    print('Saving outputs to {}'.format(outfile))
    test_graph.run_epoch(save_outputs=outfile)
    consolidate_data(outfile, include_hidden=True)


def entrypoint():
    proteins.run_commandline()


@proteins.automain
def main(_run, _config, tasks: Union[str, List[str]], model: str):
    outdir = _run.observers[0].basedir
    atexit.register(cleanup_folders, outdir)

    sess = setup_tensorflow()

    if isinstance(tasks, str):
        tasks = [tasks]

    embedding_model = ModelBuilder.build_model(model)
    task_list = TaskBuilder.build_tasks(tasks)
    task_model = TaskBuilder.build_task_model(
        embedding_model, task_list, _config['freeze_embedding_weights'])

    experiment = ProteinExperiment(task_model, task_list)

    bounds, batch_sizes = embedding_model.get_optimal_batch_sizes()
    batch_sizes = np.asarray(batch_sizes / len(tasks), np.int32)
    batch_sizes[batch_sizes <= 0] = 1
    train_data, valid_data = get_data(task_list, (bounds, batch_sizes))

    if _config['steps_per_epoch'] != -1:
        train_data = train_data.repeat()

    train_graph = rk.train.TrainGraph.from_experiment(experiment, train_data)
    test_graph = rk.train.TestGraph.from_experiment(experiment, valid_data)

    sess.run(tf.global_variables_initializer())

    print('Model Parameters: {}'.format(embedding_model.count_params()))

    if _config['load_from'] is not None:
        print('Loading weights from {}'.format(_config['load_from']))
        rk.utils.load_distributed(
            experiment.distribution_strategy, embedding_model, _config['load_from'])

    if _config['load_task_from'] is not None:
        print('Loading task weights from {}'.format(_config['load_task_from']))
        rk.utils.load_distributed(
            experiment.distribution_strategy, task_model, _config['load_task_from'])

    evaluator = MetricEvaluator(task_list[0].key_metric)

    train_graph.initialize()
    for epoch in range(_config['num_epochs']):
        train_metrics = train_graph.run_for_n_steps(_config['steps_per_epoch'], epoch_num=epoch)
        outfile = os.path.join(outdir, 'outputs.pkl') if _config['save_outputs'] else None
        test_metrics = test_graph.run_epoch(epoch_num=epoch, save_outputs=outfile)

        if all(isinstance(task, AbstractLanguageModelingTask) for task in tasks):
            with experiment.distribution_strategy.scope():
                embedding_model.save_weights('{}/epoch_{}.h5'.format(outdir, epoch), overwrite=True)

        evaluator.check_and_log_metric(train_metrics, test_metrics)

        for name, value in train_metrics.items():
            _run.log_scalar('train.{}'.format(name), value)
        for name, value in test_metrics.items():
            _run.log_scalar('valid.{}'.format(name), value)
        _run.log_scalar('runtime', round(train_metrics.runtime + test_metrics.runtime))

        if evaluator.was_improvement:
            with experiment.distribution_strategy.scope():
                embedding_model.save_weights('{}/best_weights.h5'.format(outdir, overwrite=True))
                task_model.save_weights('{}/task_weights.h5'.format(outdir, overwrite=True))
        else:
            if evaluator.n_epochs_no_improvement >= _config['patience']:
                print("Early stopping because no improvement in validation loss "
                      "for {} epochs\n".format(_config['patience']))
                break
