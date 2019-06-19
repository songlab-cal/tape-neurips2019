from typing import Dict

import tensorflow as tf
import numpy as np
import pandas as pd


def _bytes_feature(value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=value)
    )


def _int64_feature(value):
    return tf.train.Feature(
        int64_list=tf.train.Int64List(value=value)
    )


def _float_feature(value):
    return tf.train.Feature(
        float_list=tf.train.FloatList(value=value)
    )


def form_clan_fam_map(clan_fam_file: str) -> Dict[str, str]:
    data = pd.read_csv(clan_fam_file, sep='\t', na_values='\t')
    data = data.fillna('no_clan')  # Replace nans with simple string
    families = list(data.iloc[:, 0])
    clans = list(data.iloc[:, 1])
    return dict(zip(families, clans))


def to_features(**features):
    for name, value in features.items():
        if type(value) in [int, np.int32, np.int64]:
            value = _int64_feature([value])
        elif type(value) in [float, np.float32, np.float64]:
            value = _float_feature([value])
        elif type(value) == bytes:
            value = _bytes_feature([value])
        else:
            raise TypeError("Unrecognized dtype {}. Must be int, float, or bytes.".format(
                type(value)))
        features[name] = value
    features = tf.train.Features(feature=features)
    return features


def to_sequence_features(**features):
    for name, array in features.items():
        array = np.asarray(array)
        if array.ndim == 1:
            array = array[:, None]

        if array.dtype in [np.int32, np.int64]:
            array = np.asarray(array, np.int64)
            array = [_int64_feature(el) for el in array]
        elif array.dtype in [np.float32, np.float64]:
            array = np.asarray(array, np.float32)
            array = [_float_feature(el) for el in array]
        else:
            raise TypeError("Unrecognized dtype {}. Can only handle int or float dtypes.".format(array.dtype))
        features[name] = tf.train.FeatureList(feature=array)

    features = tf.train.FeatureLists(feature_list=features)
    return features
