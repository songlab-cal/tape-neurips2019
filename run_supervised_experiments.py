import os
from copy import copy
import shutil
from typing import Dict, Any, List
import argparse
from multiprocessing import Process

from sacred.observers import FileStorageObserver


def run_single_experiment(dataset: str,
                          savedir: str,
                          named_configs: List,
                          config_updates: Dict[str, Any]):
    from tape.__main__ import proteins

    config_updates.update({
        'training': {'learning_rate': 1e-4, 'use_memory_saving_gradients': True},
        'num_epochs': 1000,
        'steps_per_epoch': 200,
        'tasks': dataset})

    if not os.path.exists(savedir):
        os.mkdir(savedir)
    shutil.rmtree(proteins.observers[0].basedir)
    proteins.observers[0] = FileStorageObserver.create(
        os.path.join(savedir, dataset))

    proteins.run(
        named_configs=named_configs,
        config_updates=config_updates)


def run_supervised_experiments(load_from: str):
    if load_from is None or not os.path.isdir(load_from):
        raise FileNotFoundError("Could not find directory {}".format(load_from))

    datasets = [
        'secondary_structure_3',
        'secondary_structure_8',
        'gfp3',
        'thermostability',
        'localization'
    ]

    named_configs = []
    config_updates: Dict[str, Any] = {}

    config = os.path.join(load_from, '1', 'config.json')
    load_file = os.path.join(load_from, 'best_weights.h5')
    if os.path.exists(load_file):
        config_updates = {'load_from': load_file}
    named_configs.append(config)

    savedir = os.path.join(load_from, 'supervised')

    for dataset in datasets:
        dataset_config = copy(config_updates)
        if dataset in ['thermostability', 'localization']:
            dataset_config['gpu'] = {'device': '0'}
        p = Process(
            target=run_single_experiment,
            args=(dataset, savedir, named_configs, dataset_config))
        p.start()
        p.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run all supervised experiments')
    parser.add_argument('load_from', type=str)
    args = parser.parse_args()

    run_supervised_experiments(args.load_from)
