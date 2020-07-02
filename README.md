# DL4NLP - SharedTask

## Preperation

### Datasets

Place all datasets in ``\data``.

### Visual Embeddings

Download ``vce.normalized`` from [here](https://public.ukp.informatik.tu-darmstadt.de/naacl2019-like-humans-visual-attacks/) and place it in ``\data``.

### Adverserial Training

To run adverserial training, place the code from [Visual Attacks](https://github.com/UKPLab/naacl2019-like-humans-visual-attacks) in ``\UKP_Visual_Attacks``.
Disclaimer: While this worked flawlessly on MAC, it sometimes caused problems on Windows-Systems.

## NN

To run the best NN-system, run ``run_nn.py``.
To perform the Hyperparameter search, set ``run_best_system=False``.

## Bert

To run the best Bert-system, run ``run_bert.py``.
To perform the Hyperparameter search, set ``run_best_system=False``.