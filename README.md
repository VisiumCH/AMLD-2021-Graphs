# Shedding light on graph neural networks

This repository contains the code and slides for the _"Shedding Light on Obscure Graph Deep Learning"_ workshop presented by Visium at AMLD 2021.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VisiumCH/AMLD-2021-Graphs/blob/master/notebooks/workshop_notebook.ipynb)

## Abstract

Deep learning techniques on graphs have achieved impressive results in recent
years. Graph neural networks (GNN) combine node feature information with the
graph structure in order to make their predictions. However, this strategy
results in complex models, whose predictions can be hard to interpret.

In this workshop, you will get familiar with graph data and you will learn the
fundamental concepts behind graph neural networks. After building and training
your own GNN, you will be introduced to `GNNExplainer`, a model-agnostic framework
for interpreting GNN results. Thus, you will be able to visualize explanations
for your GNN's predictions, giving a sense to the model outputs.

## Workshop setup

Running the code of this repository is, for the most part, computationally expensive
and it is best to have GPU access. For the workshop users, we have prepared
 the notebook so that you can easily load it into Google Colab and enjoy the
 **free GPU services** offered by Google.

1) Open [Google Colab](https://colab.research.google.com/) and sign in with your
Google account or create a new one.

2) Click on `File` and then `Open notebook...`. Select the **Github** tab, look
for **VisiumCH** to find this repository and select the `workshop_notebook.ipynb`.

3) Once you are in the notebook, click on **Runtime**, **Change runtime type**
and then select **GPU** as hardware accelerator.

4) Finally, click on **Connect** and you should be ready to go!


## Requirements
For users outside the workshop who would like to experiment more with the repo outside of colab,
here are the installation steps:

Create a virtual environnement and install the dependencies:
```bash
python3 -m venv env
source env/bin/activate
xargs -a requirements.txt -L 1 pip install
```

## Basic usage

The `workshop_notebook.ipynb` contains the theory and the code to learn about
Graph Neural Networks and explainability with an hands on approach.
It contains some exercises which you shall complete in order to advance in the workshop.

Solutions are provided in `workshop_notebook_sol.ipynb`. This notebook can be
run from top to bottom.

We provide some additional code resources in the `src` module, which is
installed by the requirements files.
It implements a few more advanced methods which are used through the notebooks.

## Citations

The **GNN Explainer** algorithm we present has been developed by [Ying et al.][GNNExplainer]
If you intend to use it for anything, please consider citing the original authors in your work:

```bibtex
@misc{ying2019gnnexplainer,
  title={GNNExplainer: Generating Explanations for Graph Neural Networks},
  author={Rex Ying and Dylan Bourgeois and Jiaxuan You and Marinka Zitnik and Jure Leskovec},
  year={2019},
  eprint={1903.03894},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```

The code in this repo is adapted and corrected for the AMLD workshop.
Our implementation of the explainer is based on PyTorch Geometric.

We also borrowed the notebook structure from [PyTorch Geometric tutorials][torch-geom-tuto]
and inspired our slides on the very good [seminar from Petar
Veličković][GNN-Seminar], from which we got some figures.

[GNNExplainer]: https://arxiv.org/abs/1903.03894
[torch-geom-tuto]: https://pytorch-geometric.readthedocs.io/en/latest/notes/colabs.html
[GNN-Seminar]: https://talks.cam.ac.uk/talk/index/155341
