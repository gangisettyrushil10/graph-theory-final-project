# Graph Neural Network Link Prediction

This project explores link prediction on a real social network using a Graph Convolutional Network built with PyTorch Geometric. It trains a GCN on the Facebook social graph, evaluates predictive quality with AUC, and visualizes learned node embeddings with t-SNE to show how graph structure can be turned into meaningful latent representations.

## Highlights
- Uses the Facebook social graph dataset with 4,039 nodes and 88,234 edges.
- Trains a 2-layer GCN with PyTorch Geometric over a 100-epoch link prediction workflow.
- Evaluates model quality with AUC and generates a t-SNE embedding visualization of learned node representations.
- Packs the full experiment into 2 Python scripts, a pinned requirements file, and a bundled edge-list dataset for reproducible local runs.

## Stack
- Python
- PyTorch
- PyTorch Geometric
- NetworkX
- NumPy
- scikit-learn
- matplotlib
- pandas

## Local Setup
1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the link prediction experiment:

```bash
python GNN_Demo.py
```

3. The script will:
- Load the bundled Facebook graph dataset
- Split edges into train and test sets
- Train the GCN
- Print AUC-style evaluation output
- Render a t-SNE view of the learned embeddings

## Screenshots
- Placeholder: training output and final AUC metrics
- Placeholder: t-SNE embedding visualization
- Placeholder: graph pipeline or model architecture diagram
- Placeholder: notebook-style summary of results and observations
