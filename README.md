# PyTorch Transformer From Scratch

This repository contains a complete implementation of the Transformer architecture in [PyTorch](https://pytorch.org/).  The goal of this project is to demystify the key building blocks of the Transformer by providing readable code and clear documentation.  You can train a small Transformer model on a synthetic sequence‑to‑sequence task, visualise the training loss, and reuse the provided modules for your own research.

## Overview

The Transformer model was introduced by Vaswani et al. in the paper “Attention Is All You Need” in 2017.  Unlike earlier recurrent or convolutional sequence models, the Transformer processes all tokens in parallel using **self‑attention**.  Each encoder layer consists of a multi‑head self‑attention sub‑layer followed by a position‑wise feed‑forward network【751605239765780†L239-L245】.  Decoder layers extend this with a second attention sub‑layer that attends over the encoder outputs and apply masking so that each position can only see previous tokens【751605239765780†L296-L305】.  To inject information about token positions, sinusoidal **positional encodings** are added to the input embeddings【751605239765780†L486-L505】.  Multi‑head attention allows the model to focus on different representation subspaces at different positions【751605239765780†L374-L383】.

This project implements these components from first principles using the PyTorch `nn.Module` API:

* **Multi‑Head Attention:** computes scaled dot‑product attention for multiple heads in parallel and concatenates their outputs【751605239765780†L374-L383】.
* **Position‑wise Feed‑Forward Network:** two linear layers with a ReLU activation【751605239765780†L440-L460】.
* **Positional Encoding:** adds sine and cosine functions of different frequencies to encode token positions【751605239765780†L500-L527】.
* **Encoder and Decoder Layers:** each encoder layer contains self‑attention and feed‑forward sub‑layers with residual connections and layer normalisation.  Decoder layers add a cross‑attention sub‑layer that attends over the encoder output【751605239765780†L296-L299】【751605239765780†L374-L383】.
* **Masking:** target masks prevent a position from attending to future positions during training【751605239765780†L296-L305】.

## Project Structure

```
pytorch_transformer_from_scratch/
├── README.md             ← This file with project description and usage.
├── model.py              ← Core modules: attention, feed‑forward, positional encoding, encoder/decoder layers and Transformer.
├── train.py              ← Script to generate a toy dataset, train the model and log losses.
├── visualize.py          ← Script to visualise training and validation loss curves.
├── requirements.txt      ← Python dependencies required to run the project.
├── setup.py              ← Optional packaging file for pip installation.
└── data/
    └── synthetic.py      ← Functions to generate synthetic sequence‑to‑sequence datasets.
```

## Installation

The code has been tested with Python 3.11 and PyTorch 2.1.  We recommend creating a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The `requirements.txt` file lists PyTorch, NumPy, Matplotlib and TQDM.  If you plan to package the project or use it in your own codebase, you can install it via `pip install -e .` to enable editable mode.

## Usage

### Training on the toy dataset

The `train.py` script trains a small Transformer model on a synthetic copy task where the model must predict the input sequence shifted by one position.  The script logs training and validation loss to `training_log.csv` and saves the model to `checkpoint.pth`.  To start training with default hyperparameters, run:

```bash
python train.py
```

You can customise parameters such as the number of layers, model dimension or dataset size via command‑line arguments.  Use `python train.py --help` to see all options.

### Visualising training curves

After training, run the visualisation script to plot loss curves:

```bash
python visualize.py --logfile training_log.csv
```

This will generate a plot of the training and validation loss over epochs and save it as `loss_curve.png` in the current directory.  The code uses Matplotlib to create the plot.

### Reusing the model

The `model.py` module exposes a `Transformer` class that you can import into your own scripts.  You can use it as a starting point for tasks such as translation, summarisation or language modelling by replacing the toy dataset with real data and adjusting the vocabulary and hyperparameters accordingly.

## References

* Vaswani, A. et al., “Attention is All You Need,” *Advances in Neural Information Processing Systems*, 2017.
* The Annotated Transformer (Harvard NLP) – line‑by‑line implementation and explanation【751605239765780†L374-L383】【751605239765780†L500-L527】.
* Cross‑Entropy Loss documentation – explains the `ignore_index` parameter for ignoring padding tokens during training【565121574583199†L2973-L3040】.

We hope this project helps you understand how Transformers work under the hood.  Feel free to open issues or contribute improvements.
