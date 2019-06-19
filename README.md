# Tasks Assessing Protein Embeddings (TAPE)

Data, weights, and code for running the TAPE benchmark on a trained protein embedding. We provide a pretraining corpus, five supervised downstream tasks, pretrained language model weights, and benchmarking code.

## Paper
Preprint is available at []().

## Data

The data for training is hosted on AWS. All data are provided as TFRecords - see `tape/data_utils/` for deserializers for each dataset and documentation of data keys. If you wish to download all of TAPE, run `download_data.sh` to do so. We also provide links to each individual dataset below:

[Pretraining Corpus (Pfam)](http://s3.amazonaws.com/proteindata/data/pfam.tar.gz) __|__ [Secondary Structure](http://s3.amazonaws.com/proteindata/data/secondary_structure.tar.gz) __|__ [Contact (ProteinNet)](http://s3.amazonaws.com/proteindata/data/proteinnet.tar.gz) __|__ [Remote Homology](http://s3.amazonaws.com/proteindata/data/remote_homology.tar.gz) __|__ [Fluorescence](http://s3.amazonaws.com/proteindata/data/fluorescence.tar.gz) __|__ [Stability](http://s3.amazonaws.com/proteindata/data/stability.tar.gz)

The unsupervised Pfam dataset is around 5GB compressed and 40GB uncompressed. The supervised data is around 120MB compressed and 900 MB uncompressed. Data should be placed in the `tape/data` folder, although you may also specify a different data directory if you wish.

## Pretrained Models

We provide weights for all models pretrained as detailed in the paper. Each set of weights comes in an `h5` file and is roughly 100 MB. If you wish to download all models, run `download_pretrained_models.sh` to do so. We also provide links to each individual model's weights below:

[LSTM](http://s3.amazonaws.com/proteindata/pretrain_weights/lstm_weights.h5) __|__ [Transformer](http://s3.amazonaws.com/proteindata/pretrain_weights/transformer_weights.h5) __|__ [ResNet](http://s3.amazonaws.com/proteindata/pretrain_weights/resnet_weights.h5) __|__ [UniRep (mLSTM)](http://s3.amazonaws.com/proteindata/pretrain_weights/unirep_weights.h5) __|__ [Bepler Only Unsupervised](http://s3.amazonaws.com/proteindata/pretrain_weights/bepler_unsupervised_pretrain_weights.h5) __|__ [Bepler Unsupervised + MultiTask](http://s3.amazonaws.com/proteindata/pretrain_weights/bepler_multitask_pretrain_weights.h5)

UniRep is described in [Alley et al](https://www.biorxiv.org/content/10.1101/589333v1). Bepler refers to the models in [Bepler et al](https://openreview.net/pdf?id=SygLehCqtm).

## Code Setup

We recommend that you install `tape` into a python [virtual environment](https://virtualenv.pypa.io/en/latest/) using

`pip install -e .`

## Usage

`tape` uses [Sacred](https://sacred.readthedocs.io/en/latest/index.html) to configure and store logging information.

Sacred options are specified by running `python -m tape with <args>`. For example, to run the `transformer` model on the `masked_language_modeling` task, simply run

    python -m tape with model=transformer tasks=masked_language_modeling

Additional arguments can be specified by adding e.g. `transformer.n_layers=6`, `training.learning_rate=1e-4`, `gpu.device=0,1,2`, etc.

Global arguments are defined under `@tape.config` in `tape/__main__.py`. Model specific arguments (e.g. `transformer.n_layers`) can be found in the corresponding model file (`tape/models/Transformer.py`).

### List of Models and Tasks

The available models are:

- `transformer`
- `resnet`
- `lstm`
- `bepler`
- `unirep`
- `one_hot`
- `one_hot_evolutionary`

The available standard tasks are:

- `contact_map`
- `stability`
- `fluorescence`
- `language_modeling`
- `masked_language_modeling`
- `remote_homology`
- `secondary_structure`

Additionally, we have some Unirep and Bepler specific tasks:

- `bepler_language_modeling`: for unsupervised pre-training of the Bepler model.
- `unidirectional_language_modeling`: for unsupervised pre-training of the Unirep model.

Finally we also provide the `netsurf` task, which does the full multi-task Netsurf training described in the original [paper](https://www.biorxiv.org/content/10.1101/311209v1). This is done on the same dataset as secondary structure.

The available models and tasks can be found in `tape/models/ModelBuilder.py` and `tape/tasks/TaskBuilder.py`.

## Loading a Model

There are two ways of loading a model, depending on whether you want to load the unsupervised pre-training weights or the supervised task-specific weights. Loading unsupervised weights is done by passing the argument `load_from=</path/to/unsupervised_weights.h5>`. Loading supervised weights is done by passing the argument `load_task_from=</path/to/supervised_weights.h5>`.

## Saving Results

Results will be stored in `tape/results`. Each run will be placed in a timestamped directory. All `tape` sources will automatically be saved, along with the config and per-epoch metrics.

## Leaderboard

We will soon have a leaderboard available for tracking progress on the core five TAPE tasks, so check back for a link here. See the main tables in our paper for a sense of where performance stands at this point. Publication on the leaderboard will be contingent on meeting the following citation guidelines.

In the meantime, here's a temporary leaderboard for each task. All reported models on this leaderboard use unsupervised pretraining.

### Secondary Structure

| Ranking | Model | Accuracy (3-class) |
|:-:|:-:|:-:|
| 1. | One Hot + Alignment | 0.80 |
| 2. | LSTM | 0.75 |
| 2. | ResNet | 0.75 |
| 4. | Transformer | 0.73 |
| 4. | Bepler | 0.73 |
| 4. | Unirep | 0.73 |
| 7. | One Hot | 0.69 |

### Contact Prediction

| Ranking | Model | L/5 Medium + Long Range |
|:-:|:-:|:-:|
| 1. | One Hot + Alignment | 0.64 |
| 2. | LSTM | 0.49 |
| 3. | Unirep | 0.45 |
| 4. | Bepler | 0.42 |
| 5. | Transformer | 0.34 |
| 6. | One Hot | 0.29 |
| 7. | ResNet | 0.28 |

### Remote Homology Detection

| Ranking | Model | Top 1 Accuracy |
|:-:|:-:|:-:|
| 1. | LSTM | 0.26 |
| 2. | Unirep | 0.23 |
| 3. | Transformer | 0.21 |
| 4. | Bepler | 0.17 |
| 4. | ResNet | 0.17 |
| 6. | One Hot + Alignment | 0.09 |
| 6. | One Hot | 0.09 |

### Fluorescence

| Ranking | Model | Spearman's rho |
|:-:|:-:|:-:|
| 1. | Transformer | 0.68 |
| 2. | LSTM | 0.67 |
| 2. | Unirep | 0.67 |
| 4. | Bepler | 0.33 |
| 5. | ResNet | 0.21 |
| 6. | One Hot | 0.14 |

### Stability

| Ranking | Model | Spearman's rho |
|:-:|:-:|:-:|
| 1. | Transformer | 0.73 |
| 1. | Unirep | 0.73 |
| 1. | ResNet | 0.73 |
| 4. | LSTM | 0.69 |
| 5. | Bepler | 0.64 |
| 6. | One Hot | 0.19 |

## Citation Guidelines

If you find TAPE useful, please cite our corresponding paper. Additionally, __anyone using the datasets provided in TAPE must describe and cite all dataset components they use__. Producing these data is time and resource intensive, and we insist this be recognized by all TAPE users. For convenience,`data_refs.bib` contains all necessary citations. We also provide each individual citation below.

__TAPE (Our paper):__
```
@article{
author = {Rao, Roshan and Bhattacharya, Nicholas and Thomas, Neil and Duan, Yan and Chen, Xi and Canny, John and Abbeel, Pieter and Song, Yun S},
title = {Evaluating Protein Transfer Learning with TAPE},
year = {2019}
}
```

__Pfam (Pretraining):__
```
@article{
author = {El-Gebali, Sara and Mistry, Jaina and Bateman, Alex and Eddy, Sean R and Luciani, Aur{\'{e}}lien and Potter, Simon C and Qureshi, Matloob and Richardson, Lorna J and Salazar, Gustavo A and Smart, Alfredo and Sonnhammer, Erik L L and Hirsh, Layla and Paladin, Lisanna and Piovesan, Damiano and Tosatto, Silvio C E and Finn, Robert D},
doi = {10.1093/nar/gky995},
file = {::},
issn = {0305-1048},
journal = {Nucleic Acids Research},
keywords = {community,protein domains,tandem repeat sequences},
number = {D1},
pages = {D427--D432},
publisher = {Narnia},
title = {{The Pfam protein families database in 2019}},
url = {https://academic.oup.com/nar/article/47/D1/D427/5144153},
volume = {47},
year = {2019}
}
```
__SCOPe: (Remote Homology and Contact)__-
```
@article{
  title={SCOPe: Structural Classification of Proteins—extended, integrating SCOP and ASTRAL data and classification of new structures},
  author={Fox, Naomi K and Brenner, Steven E and Chandonia, John-Marc},
  journal={Nucleic acids research},
  volume={42},
  number={D1},
  pages={D304--D309},
  year={2013},
  publisher={Oxford University Press}
}
```
__PDB: (Secondary Structure and Contact)__
```
@article{
  title={The protein data bank},
  author={Berman, Helen M and Westbrook, John and Feng, Zukang and Gilliland, Gary and Bhat, Talapady N and Weissig, Helge and Shindyalov, Ilya N and Bourne, Philip E},
  journal={Nucleic acids research},
  volume={28},
  number={1},
  pages={235--242},
  year={2000},
  publisher={Oxford University Press}
}
```

__CASP12: (Secondary Structure and Contact)__
```
@article{
author = {Moult, John and Fidelis, Krzysztof and Kryshtafovych, Andriy and Schwede, Torsten and Tramontano, Anna},
doi = {10.1002/prot.25415},
issn = {08873585},
journal = {Proteins: Structure, Function, and Bioinformatics},
keywords = {CASP,community wide experiment,protein structure prediction},
pages = {7--15},
publisher = {John Wiley {\&} Sons, Ltd},
title = {{Critical assessment of methods of protein structure prediction (CASP)-Round XII}},
url = {http://doi.wiley.com/10.1002/prot.25415},
volume = {86},
year = {2018}
}
```

__ProteinNet: (Contact)__
```
@article{
  title={ProteinNet: a standardized data set for machine learning of protein structure},
  author={AlQuraishi, Mohammed},
  journal={arXiv preprint arXiv:1902.00249},
  year={2019}
}
```

__Fluorescence:__
```
@article{
  title={Local fitness landscape of the green fluorescent protein},
  author={Sarkisyan, Karen S and Bolotin, Dmitry A and Meer, Margarita V and Usmanova, Dinara R and Mishin, Alexander S and Sharonov, George V and Ivankov, Dmitry N and Bozhanova, Nina G and Baranov, Mikhail S and Soylemez, Onuralp and others},
  journal={Nature},
  volume={533},
  number={7603},
  pages={397},
  year={2016},
  publisher={Nature Publishing Group}
}
```

__Stability:__
```
@article{
  title={Global analysis of protein folding using massively parallel design, synthesis, and testing},
  author={Rocklin, Gabriel J and Chidyausiku, Tamuka M and Goreshnik, Inna and Ford, Alex and Houliston, Scott and Lemak, Alexander and Carter, Lauren and Ravichandran, Rashmi and Mulligan, Vikram K and Chevalier, Aaron and others},
  journal={Science},
  volume={357},
  number={6347},
  pages={168--175},
  year={2017},
  publisher={American Association for the Advancement of Science}
}
```
