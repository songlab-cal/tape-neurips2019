"""Defines a simple model consisting of a single embedding layer and a
1D convolutionn that can be added to tape. This file can then be run directly with

    python simple_model.py with model=my_simple_model tasks=<task_name>

Just make sure you specify the data directory correctly if the data is not in `./data`.
"""

from typing import Tuple, List
from tensorflow.keras.layers import Embedding, Conv1D
import numpy as np

from tape.models import AbstractTapeModel
from tape.models import ModelBuilder


class MySimpleModel(AbstractTapeModel):

    def __init__(self,
                 n_symbols: int,  # This argument is required!
                 filters: int = 32  # There's no way to change this
                                    # from the commandline - see `my_simple_model_with_hparams.py`
                 ) -> None:
        super().__init__(n_symbols)

        self.input_embedding = Embedding(n_symbols, 10)
        self.conv1d = Conv1D(filters=filters, kernel_size=7, strides=1, padding='same')

    def call(self, inputs):
        """
        Args:
            inputs dictionary containing:
                sequence: tf.Tensor[int32] - Amino acid sequence,
                    a padded tensor with shape [batch_size, MAX_PROTEIN_LENGTH]
                protein_length: tf.Tensor[int32] - Length of each protein in the sequence,
                    a tensor with shape [batch_size]

        Output:
            added to the inputs dictionary:
                encoder_output: tf.Tensor[float32] - embedding of each amino acid
                    a tensor with shape [batch_size, MAX_PROTEIN_LENGTH, 32]
        """

        sequence = inputs['primary']

        # protein_length not used here, but useful if you're going to do padding,
        # sequence masking, etc.
        # protein_length = inputs['protein_length']

        # Embed the sequence into a 10-dimensional vector space
        embedded = self.input_embedding(sequence)

        # Use a single 1D convolution
        encoder_output = self.conv1d(embedded)

        # Add the result to the dictionary
        inputs['encoder_output'] = encoder_output

        # Return the dictionary
        return inputs

    def get_optimal_batch_sizes(self) -> Tuple[List[int], List[int]]:
        """Need to figure out what batch sizes to use for different sequence lengths.
        You can make this pretty fine-grained but here we're just going to say use
        a batch of 64 for sequences of length < 500, a batch of 32 for sequences of length
        < 1000, a batch of 16 for sequences of length < 1500, a batch of 8 for sequences
        of length < 2000, and a batch of 4 for anything longer"""

        # Define the bucket sizes we care about
        bucket_sizes = [500, 1000, 1500, 2000]

        # Define the batch sizes we care about
        # (1 more than bucket sizes to handle things larger than largest bucket size)
        batch_sizes = [64, 32, 16, 8, 4]

        return np.array(bucket_sizes), np.array(batch_sizes)


# Register the model
ModelBuilder.add_model('my_simple_model', MySimpleModel)


if __name__ == '__main__':
    # Run tape from this file! This will ensure that the below model registration code
    # is run. If you want to run tape normally, you'll need to modify the code to register
    # your model from within tape.
    from tape.__main__ import proteins
    proteins.run_commandline()
