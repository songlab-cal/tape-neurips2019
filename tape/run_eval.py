from tape.analysis import get_config
import os

from tape.__main__ import proteins

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('outdir')
    parser.add_argument('--datafile', default='')
    args = parser.parse_args()

    config = get_config(args.outdir)
    task = config['tasks']

    if isinstance(task, (tuple, list)):
        task = task[0]

    config_updates = {
        'tasks': task,
        'load_task_from': os.path.join(args.outdir, 'task_weights.h5'),
        'save_outputs': True,
        'datafile': args.datafile}

    proteins.run(
        'eval',
        named_configs=[os.path.join(args.outdir, '1', 'config.json')],
        config_updates=config_updates,
    )


if __name__ == '__main__':
    main()
