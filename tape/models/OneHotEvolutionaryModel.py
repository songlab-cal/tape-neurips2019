import tensorflow.keras.backend as K

from .OneHotModel import OneHotModel


class OneHotEvolutionaryModel(OneHotModel):

    def call(self, inputs):
        encoder_output = K.one_hot(inputs['primary'], self._n_symbols)
        try: 
            encoder_output = K.concatenate((encoder_output, inputs['evolutionary']))
        except KeyError:
            raise TypeError("Evolutionary inputs not available for this task.")
        inputs['encoder_output'] = encoder_output
        return inputs
