from tensorflow.keras import Model
from tensorflow.keras.layers import Conv1D
from rinokeras.layers import LayerNorm, Stack
from rinokeras.layers import WeightNormDense as Dense


class AminoAcidClassPredictor(Model):

    def __init__(self,
                 n_classes: int,
                 input_name: str = 'encoder_output',
                 output_name: str = 'sequence_logits',
                 use_conv: bool = True) -> None:
        super().__init__()
        self._input_name = input_name
        self._output_name = output_name
        if use_conv:
            self.predict_class = Stack([
                LayerNorm(),
                Conv1D(128, 5, activation='relu', padding='same', use_bias=True),
                Conv1D(n_classes, 3, activation=None, padding='same', use_bias=True)])
        else:
            self.predict_class = Stack([
                LayerNorm(),
                Dense(512, activation='relu'),
                Dense(n_classes, activation=None)])

    def call(self, inputs):
        sequence_logits = self.predict_class(inputs[self._input_name])

        inputs[self._output_name] = sequence_logits
        return inputs
