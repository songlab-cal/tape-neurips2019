from typing import Dict, Type, List
from sacred import Ingredient
from tensorflow.keras import Model

from tape.data_utils import PFAM_VOCAB

from .Transformer import Transformer, transformer_hparams
from .Resnet import Resnet, resnet_hparams
from .BidirectionalLSTM import BidirectionalLSTM, lstm_hparams
from .BeplerModel import BeplerModel, bepler_hparams
from .UniRepModel import UniRepModel, unirep_hparams
from .OneHotModel import OneHotModel
from .OneHotEvolutionaryModel import OneHotEvolutionaryModel


class ModelBuilder:

    models: Dict[str, Type[Model]] = {
        'transformer': Transformer,
        'resnet': Resnet,
        'lstm': BidirectionalLSTM,
        'bepler': BeplerModel,
        'unirep': UniRepModel,
        'one_hot': OneHotModel,
        'one_hot_evolutionary': OneHotEvolutionaryModel}

    hparams: List[Ingredient] = [
        transformer_hparams,
        resnet_hparams,
        lstm_hparams,
        bepler_hparams,
        unirep_hparams]

    @staticmethod
    def build_model(model_name: str) -> Model:
        if model_name.lower() == 'bidirectional_lstm':
            model_name = 'lstm'

        n_symbols = len(PFAM_VOCAB)
        model_type = ModelBuilder.models[model_name.lower()]
        return model_type(n_symbols)

    @staticmethod
    def add_model(model_name: str, model: Type[Model], hparams: Ingredient) -> None:
        assert isinstance(model, type) and isinstance(hparams, Ingredient)
        ModelBuilder.models[model_name] = model
        ModelBuilder.hparams.append(hparams)
