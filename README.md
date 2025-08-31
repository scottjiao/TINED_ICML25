
# TINED


## Preparing datasets
To run experiments for dataset used in the paper, please download from the following links and put them under `data/` (see below for instructions on organizing the datasets).

- *CPF data* (`cora`, `citeseer`, `pubmed`, `a-computer`, and `a-photo`): Download the '.npz' files from [here](https://github.com/BUPT-GAMMA/CPF/tree/master/data/npz). Rename `amazon_electronics_computers.npz` and `amazon_electronics_photo.npz` to `a-computer.npz` and `a-photo.npz` respectively.

- *OGB data* (`ogbn-arxiv` and `ogbn-products`): Datasets will be automatically downloaded when running the `load_data` function in `dataloader.py`. More details [here](https://ogb.stanford.edu/).


## Usage



1. run teachers by `./tasks/1_run_all_teacher.py`, the script will automatically save all the layer and embedding information of the learned teacher in `./results/outputs_with_learned_MLP`

2. compute the Dirichlet Energy information of teachers to guide the student by running `./tasks/2_compute_and_record_teacher_DERs.py`

3. run TINED by `./tasks/3_run_all_TINED.py`, you can find the results in `./results/rep_all/rep_all_TINED`. For example, for transductive cora, you can find the results in `./results/rep_all_TINED/transductive/cora/SAGE_MLP_from_sequence_of_layers/exp_results.csv`

3. run TINED+ by `./tasks/4_run_all_TINED+.py`, you can find the results in `./results/rep_all/rep_all_TINED+`. For example, for transductive cora, you can find the results in `./results/rep_all_TINED+/transductive/cora/SAGE_MLP_from_sequence_of_layers/exp_results.csv`

## Core part of codes for reviewing:

1. class "MLP_from_sequence_of_layers" in `models.py` is the core part of the model, which is the implementation of the TINED base model for teacher injection.
2. function "DE_regularization" in `train_and_eval.py` implements the Dirichlet Energy Distillation part of our paper.