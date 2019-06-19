from tape.data_utils import deserialize_secondary_structure
from .Task import SequenceToSequenceClassificationTask
from sacred import Ingredient

secondary_structure_params = Ingredient('secondary_structure')


@secondary_structure_params.config
def secondary_structure_config():
    num_classes = 3  # noqa: F841


class SecondaryStructureTask(SequenceToSequenceClassificationTask):

    def __init__(self, num_classes: int = 8):
        assert num_classes in [3, 8]
        super().__init__(
            key_metric='ACC',
            deserialization_func=deserialize_secondary_structure,
            n_classes=num_classes,
            label_name='ss{}'.format(num_classes),
            input_name='encoder_output',
            output_name='sequence_logits')
