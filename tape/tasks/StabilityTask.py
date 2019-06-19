from tape.data_utils import deserialize_stability_sequence
from .Task import SequenceToFloatTask


class StabilityTask(SequenceToFloatTask):

    def __init__(self):
        d_output = 1
        super().__init__(
            key_metric='MAE',
            deserialization_func=deserialize_stability_sequence,
            d_output=d_output,
            label='stability_score',
            input_name='encoder_output',
            output_name='prediction')
