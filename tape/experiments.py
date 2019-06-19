from typing import Union, Callable, Tuple, List, Type

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.contrib.distribute import DistributionStrategy, MirroredStrategy
from tensorflow.keras import Model
from sacred import Ingredient
from rinokeras.train import Experiment
from rinokeras.utils import PiecewiseSchedule

from tape.tasks import Task

training_params = Ingredient('training')


@training_params.config
def training_cfg():
    """Configure the rinokeras Experiment"""
    optimizer = 'adam'  # noqa: F841
    learning_rate = 1e-3  # noqa: F841
    gradient_clipping = 'norm'  # noqa: F841
    gradient_clipping_bounds = 1  # noqa: F841
    use_memory_saving_gradients = False  # noqa: F841


class ProteinExperiment(Experiment):

    @training_params.capture
    def __init__(self,
                 model: Model,
                 tasks: List[Task],
                 optimizer: str = 'adam',
                 learning_rate: Union[float, Callable[[int], float]] = 1e-3,
                 gradient_clipping: str = 'norm',
                 gradient_clipping_bounds: Union[float, Tuple[float, float]] = 1.,
                 return_loss_summaries: bool = False,
                 return_variable_summaries: bool = False,
                 return_grad_summaries: bool = False,
                 distribution_strategy: Type[DistributionStrategy] = MirroredStrategy,
                 use_memory_saving_gradients: bool = False) -> None:

        learning_rate_func = PiecewiseSchedule(
            [(0, 1e-6),
             (100, 1e-4),
             (1000, learning_rate)],
            outside_value=learning_rate)

        super().__init__(
            model, optimizer, learning_rate_func.value, gradient_clipping, gradient_clipping_bounds,
            return_loss_summaries, return_variable_summaries, return_grad_summaries,
            distribution_strategy(), use_memory_saving_gradients)
        self._tasks = tasks

    def build_model(self, inputs):
        if len(self._tasks) == 1:
            inputs = (inputs,)

        return tuple(self.model(inp) for inp in inputs)

    def loss_function(self, inputs, outputs):
        total_loss = 0
        total_metrics = {}

        if len(self._tasks) == 1:
            inputs = (inputs,)

        try:
            batch_size = K.cast(tf.shape(inputs[0]['primary'])[0], K.floatx())
        except KeyError:
            batch_size = K.cast(tf.shape(inputs[0]['first']['primary'])[0], K.floatx())

        for input_, output, task in zip(inputs, outputs, self._tasks):
            loss, metrics = task.loss_function(input_, output)
            loss = tf.check_numerics(loss, 'loss from {}'.format(task.__class__.__name__))
            for name, value in metrics.items():
                metrics[name] = tf.check_numerics(value, 'metric {}'.format(name))
            total_loss += loss
            total_metrics.update(metrics)

        total_loss *= batch_size
        for key in total_metrics:
            total_metrics[key] *= batch_size
        total_metrics['batch_size'] = batch_size

        return total_loss, total_metrics
