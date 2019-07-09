from tape.__main__ import proteins

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile', default='')
    parser.add_argument('--model', default='')
    args = parser.parse_args()

    task = 'embedding'

    config_updates = {
        'tasks': task,
        'save_outputs': True,
        'datafile': args.datafile,
        'model': args.model}

    proteins.run(
        'embed',
        config_updates=config_updates,
    )


if __name__ == '__main__':
    main()
