from tape.data_utils import deserialize_fluorescence_sequence
from .Task import SequenceToFloatTask


class FluorescenceTask(SequenceToFloatTask):

    def __init__(self):
        d_output = 1
        super().__init__(
            key_metric='MAE',
            deserialization_func=deserialize_fluorescence_sequence,
            d_output=d_output,
            label='log_fluorescence',
            input_name='encoder_output',
            output_name='prediction')
