# Exploiting the Structure of Two Graphs with Graph Neural Networks

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/liuyuan999/Penalty_Based_Lagrangian_Bilevel_Tianyi_Chen-s_Lab/blob/main/LICENSE) [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)  [![Arxiv link](https://img.shields.io/badge/cs.LG-2411.05119-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2411.05119)

This repository contains the implementation for the paper ["Exploiting the Structure of Two Graphs with Graph Neural Networks"](https://arxiv.org/abs/2411.05119).

## Overview

The repository provides code and experiments for leveraging the structure of two graphs using various Graph Neural Network (GNN) architectures. It includes implementations of models such as IOGCN, IOGAT, IOMLP, and GCNH, along with utilities for graph processing and mesh generation.

## Results

- **Root folder**: Results for the experiments are available in the respective Jupyter notebooks:
  - `SelectionNodesGxGy.ipynb`: results of the experiment in Section V.B about subgraph Feature estimation.
  - `CCA-Self-Supervised-Subgraph.ipynb`: results of the experiment in Section V.E about self-supervised learning, using the framework in Section IV of the paper.
  - `RecommenderSystemLF.ipynb`: results of the experiment in section V.F about recommender systems using Latent Factor models.
- **Interpolation Results**: The results related to interpolation experiments can be found in the `interpolation/` folder. More precisely:
  - `image-segmentation.ipynb`: results of the experiment in Section V.C about image interpolation
  - `Airfoil_GNN.ipynb`: results of the experiment in Section V.D about fluid flow predition.

## Citation

If you use this code, please cite the paper:

```
@article{tenorio24iognn,
    title={Exploiting the Structure of Two Graphs with Graph Neural Networks},
    author={Victor M. Tenorio, Antonio G. Marques},
    journal={arXiv preprint arXiv:2411.05119},
    year={2024}
}
```
