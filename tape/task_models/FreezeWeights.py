import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer


class FreezeWeights(Layer):

    def call(self, inputs):
        inputs['encoder_output'] = K.stop_gradient(inputs['encoder_output'])
        return inputs
