# -*- coding: utf-8 -*-
from setuptools import setup


with open('README.md', 'r') as rf:
    README = rf.read()

with open('LICENSE', 'r') as lf:
    LICENSE = lf.read()

setup(
    name='tape',
    version='0.1',
    description='Protein Benchmarking Repository',
    long_description=README,
    author='Roshan Rao, Nick Bhattacharya, Neil Thomas',
    author_email='roshan_rao@berkeley.edu, nickbhat@berkeley.edu, nthomas@berkeley.edu',
    url='https://github.com/nickbhat/tape',
    license=LICENSE,
    install_requires=[
        'tensorflow-gpu<1.14.0',  # https://github.com/IDSIA/sacred/issues/493
        'numpy',
        'rinokeras==1.1.1',
        'biopython',
        'sacred',
        'table_logger',
        'pandas',
        'pillow',
        'pymongo'],   # https://github.com/IDSIA/sacred/issues/491]
    entry_points={
        'console_scripts': [
            'tape = tape.__main__:entrypoint',
            'tape-eval = tape.run_eval:main',
            'tape-supervised = tape.run_supervised_experiments:main',
        ],
    },
)
