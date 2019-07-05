"""Building off of simple_model.py, this adds hyperparameters to the model.

Usage:
    python simple_model_with_hparams.py with model=my_simple_model_with_hparams tasks=<task_name>

We'll also add a hyperparameter object called `my_hparams`. This allows you to set the number of filters like so:

    python simple_model_with_hparams.py with model=my_simple_model_with_hparams my_hparams.filters=<filter_num> tasks=<task_name>
"""""


from tensorflow.keras.layers import Embedding, Conv1D
from sacred import Ingredient
from tape.models import ModelBuilder

# Because we registered MySimpleModel in the body of `simple_model.py`, this import statement
# will run that registration code, so we'll also be able to run with `model=my_simple_model`.
from simple_model import MySimpleModel


hparams = Ingredient('my_hparams')


@hparams.config
def model_cfg():
    filters = 32  # must have the same name as the hyperparameters below


class MySimpleModelWithHparams(MySimpleModel):

    @hparams.capture
    def __init__(self,
                 n_symbols: int,  # This argument is required!
                 filters: int = 32  # This gets captured by Sacred so you can pass values in
                 ) -> None:
        super().__init__(n_symbols, filters)
        print("Creating Model with {} filters".format(self.conv1d.filters))


# Register the model and hparams
ModelBuilder.add_model('my_simple_model_with_hparams', MySimpleModelWithHparams, hparams)


if __name__ == '__main__':
    # Run tape from this file! This will ensure that the below model registration code
    # is run. If you want to run tape normally, you'll need to modify the code to register
    # your model from within tape.
    from tape.__main__ import proteins
    proteins.run_commandline()
