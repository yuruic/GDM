# GDM: Graph Distribution Matching

This repository contains the implementation of Graph Distribution Matching (GDM) for globally interpretable graph learning, as presented in the paper "Globally Interpretable Graph Learning via Distribution Matching" (WWW '24).

## Overview

Graph Distribution Matching (GDM) provides global interpretation for graph neural networks by distilling high-level patterns that dominate the learning procedure. Unlike local interpretation methods, GDM captures model behavior across instances by matching distributions between original and interpretive graphs in the GNN's feature space during training.

The method includes a novel model fidelity metric and demonstrates high accuracy, efficiency, and the ability to reveal class-relevant structures across multiple graph classification datasets.

## Features

- **Global Interpretation**: High-level patterns that dominate graph learning
- **Distribution Matching**: Matches distributions between original and interpretive graphs
- **Model Fidelity**: Novel metric for evaluating interpretation quality
- **Multiple Datasets**: BA_2Motifs, BA_Community, BA_shapes, Graph-SST5...

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yuruic/GDM.git
cd GDM
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the main training script:
```bash
python main.py --DATASET MUTAG
```

### Configuration

Modify `config.json` to adjust training parameters, model architecture, and dataset settings.

### Data Loading

The `load_data.py` module handles dataset loading and preprocessing. Supported datasets are automatically downloaded and processed.

## Project Structure

```
GDM/
├── main.py              # Main training script
├── models.py            # Model definitions
├── models_gcn.py        # GCN-based model implementations
├── load_data.py         # Data loading utilities
├── utils.py             # Utility functions
├── config.json          # Configuration file
├── requirements.txt     # Python dependencies
└── datasets/            # Dataset directory
    ├── BA_2Motifs/
    ├── Graph-SST5/
    └── ...              # Other datasets
```

## Model Architecture

The implementation includes:
- **Graph Neural Networks**: Various GNN architectures for graph representation learning
- **Distribution Matching**: Core algorithm that matches distributions between original and interpretive graphs
- **Model Fidelity Evaluation**: Novel metric for assessing interpretation quality
- **Training Pipeline**: End-to-end training with configurable parameters for both original and interpretive models

## Training

1. **Configuration**: Set your desired parameters in `config.json`
2. **Data Preparation**: Ensure your target dataset is available in the `datasets/` directory
3. **Training**: Run `python main.py` to start training

## Citation

If you use this code in your research, please cite the original paper:

```bibtex
@inproceedings{Nian_2024, series={WWW ’24},
   title={Globally Interpretable Graph Learning via Distribution Matching},
   url={http://dx.doi.org/10.1145/3589334.3645674},
   DOI={10.1145/3589334.3645674},
   booktitle={Proceedings of the ACM Web Conference 2024},
   publisher={ACM},
   author={Nian, Yi and Chang, Yurui and Jin, Wei and Lin, Lu},
   year={2024},
   month=may, pages={992–1002},
   collection={WWW ’24} }
```
