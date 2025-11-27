# Graph Theory Final Project

Link prediction on social networks using Graph Neural Networks (GNNs).

This project implements a Graph Convolutional Network (GCN) for predicting missing links in the Facebook social network dataset. The model learns node embeddings and uses them to score potential edges, achieving link prediction through a binary classification approach.

## What It Does

The code trains a GCN to learn meaningful representations of nodes in the Facebook network. Given a set of training edges, it learns to distinguish between real connections and non-existent ones. The model is evaluated using AUC score on a held-out test set.

## Files

- `GNN_Demo.py` - Main implementation with GCN model, training loop, and evaluation
- `ML_example.py` - Simple linear regression example (separate demo)
- `facebook_combined.txt` - Facebook social network edge list dataset

## Setup

Install the required dependencies:

```bash
pip install torch torch-geometric networkx numpy matplotlib scikit-learn pandas
```

## Running

Make sure you have the `facebook_combined.txt` file in the same directory, then run:

```bash
python GNN_Demo.py
```

The script will:
1. Load the Facebook network dataset
2. Split edges into train/test sets
3. Train a 2-layer GCN for 100 epochs
4. Evaluate on test set and print AUC score
5. Generate a t-SNE visualization of learned node embeddings

## Model Architecture

- **Input**: Node features (initialized as ones) and edge connectivity
- **Encoder**: 2-layer GCN with ReLU activation and dropout
  - Layer 1: 1 → 128 hidden channels
  - Layer 2: 128 → 64 output channels
- **Decoder**: Dot product between node embeddings
- **Loss**: Binary cross-entropy with logits

## Results

The model achieves link prediction performance measured by AUC score on the test set. Training loss is printed every 10 epochs, and final test AUC is displayed at the end.

