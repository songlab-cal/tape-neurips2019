from typing import Dict, Optional, Any, Union, List
import operator
import os
import json


def get_config(directory: str) -> Dict[str, Any]:
    with open(os.path.join(directory, '1', 'config.json')) as f:
        return json.load(f)


def get_name(config: Dict[str, Any]) -> str:
    model_config = config['model']
    model_type = model_config['model_type']
    if model_type == 'bert':
        mask_type = model_config['mask_type']
        mask_percentage = str(model_config['mask_percentage'])
        return '_'.join([model_type, mask_type, mask_percentage])
    else:
        return model_type


def get_parent(config: Dict[str, Any]) -> Optional[str]:
    if config['load_from'] is None:
        return None
    return config['load_from'].rsplit('/', maxsplit=1)[0]


def get_parent_name(config: Dict[str, Any]) -> str:
    parent = get_parent(config)
    if parent is None:
        return config['model']['model_type'] + '_no-pretrain'
    else:
        return get_name(get_config(parent))


def get_best_metric(directory: str, trial: Union[int, str] = '1') -> float:
    with open(os.path.join(directory, str(trial), 'metrics.json')) as f:
        metrics = json.load(f)

    for metric in metrics:
        if metric[:5] == 'valid' and metric != 'valid.Loss':
            best = metrics[metric]['values']
            key_metric = metric

    if 'acc' in key_metric.lower():
        best = max(best)
    else:
        best = min(best)
    return best


def get_best_loss(directory: str, trial: Union[int, str] = '1') -> float:
    with open(os.path.join(directory, str(trial), 'metrics.json')) as f:
        metrics = json.load(f)

    best = metrics['valid.Loss']['values']
    best = min(best)
    return best


def get_loss_across_trials(directory: str) -> List[float]:
    directories = os.listdir(directory)
    valid = filter(operator.methodcaller('isnumeric'), directories)
    sorted_valid = sorted(valid, key=int)
    return [get_best_loss(directory, trial) for trial in sorted_valid]


def get_metric_across_trials(directory: str) -> List[float]:
    directories = os.listdir(directory)
    valid = filter(operator.methodcaller('isnumeric'), directories)
    sorted_valid = sorted(valid, key=int)
    return [get_best_metric(directory, trial) for trial in sorted_valid]
