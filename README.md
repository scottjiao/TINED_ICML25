
# TINED: GNNs-to-MLPs by Teacher Injection and Dirichlet Energy Distillation (ICML 2025)

## Code and data for our Heterogeneous Graph Neural Network method TINED: GNNs-to-MLPs by Teacher Injection and Dirichlet Energy Distillation (ICML 2025) (https://openreview.net/pdf?id=nshtqLv4r4)

## Abstract
Graph Neural Networks (GNNs) are pivotal in graph-based learning, particularly excelling in node classification. However, their scalability is hindered by the need for multi-hop data during inference, limiting their application in latency-sensitive scenarios. Recent efforts to distill GNNs into multi-layer perceptrons (MLPs) for faster inference often underutilize the layer-level insights of GNNs. In this paper, we present TINED, a novel approach that distills GNNs to MLPs on a layer-by-layer basis using Teacher Injection and Dirichlet Energy Distillation techniques. We focus on two key operations in GNN layers: feature transformation (FT) and graph propagation (GP). We recognize that FT is computationally equivalent to a fully-connected (FC) layer in MLPs. Thus, we propose directly transferring teacher parameters from an FT in a GNN to an FC layer in the student MLP, enhanced by fine-tuning. In TINED, the FC layers in an MLP replicate the sequence of FTs and GPs in the GNN. We also establish a theoretical bound for GP approximation. Furthermore, we note that FT and GP operations in GNN layers often exhibit opposing smoothing effects: GP is aggressive, while FT is conservative. Using Dirichlet energy, we develop a DE ratio to measure these effects and propose Dirichlet Energy Distillation to convey these characteristics from GNN layers to MLP layers. Extensive experiments show that TINED outperforms GNNs and leading distillation methods across various settings and seven datasets. Source code are available at https://github.com/scottjiao/TINED_ICML25/.

Please cite our paper if you use the code or data.

```@InProceedings{icml-zhou25,
  title = 	 {TINED: GNNs-to-MLPs by Teacher Injection and Dirichlet Energy Distillation},
  author =       {Ziang Zhou, Zhihao Ding, Jieming Shi, Li Qing, Shiqi Shen},
  booktitle = 	 {International Conference on Machine Learning},
  year = 	 {2025},
}
```


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
