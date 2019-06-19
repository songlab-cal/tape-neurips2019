import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Dropout
import rinokeras as rk
from rinokeras.layers import WeightNormDense as Dense
from rinokeras.layers import LayerNorm, Stack

from .layers import ComputeClassVector


class GlobalVectorPredictor(Model):

    def __init__(self,
                 d_output: int,
                 input_name: str = 'cls_vector',
                 output_name: str = 'prediction') -> None:
        super().__init__()
        self._d_output = d_output
        self._input_name = input_name
        self._output_name = output_name
        self.predict_vector = Stack([LayerNorm(), Dense(512, 'relu'), Dropout(0.5), Dense(d_output)])

    def call(self, inputs):
        prediction = self.predict_vector(inputs[self._input_name])
        inputs[self._output_name] = prediction

        return inputs
