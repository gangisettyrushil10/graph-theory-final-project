import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score

# Step 1: Load the dataset
G = nx.read_edgelist('facebook_combined.txt', nodetype=int)

# Convert the NetworkX graph to PyTorch Geometric format
edge_index = torch.tensor(np.array(list(G.edges())).T, dtype=torch.long)
num_nodes = len(G.nodes)
node_features = torch.ones((num_nodes, 1), dtype=torch.float)

# Create the PyTorch Geometric data object
data = Data(x=node_features, edge_index=edge_index)

# Step 2: Split the edges into train and test sets for link prediction
data = train_test_split_edges(data)

class GCNLinkPrediction(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNLinkPrediction, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_index):
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        return z

# Initialize the model and optimizer
model = GCNLinkPrediction(in_channels=1, hidden_channels=128, out_channels=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def get_link_labels(pos_edge_index, neg_edge_index):
    num_links = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(num_links, dtype=torch.float)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    z = model(data.x, data.train_pos_edge_index)
    
    # Positive edges
    pos_score = model.decode(z, data.train_pos_edge_index)
    
    # Generate negative edges
    neg_edge_index = data.train_pos_edge_index[:, torch.randperm(data.train_pos_edge_index.size(1))]
    neg_score = model.decode(z, neg_edge_index)
    
    scores = torch.cat([pos_score, neg_score], dim=0)
    labels = get_link_labels(data.train_pos_edge_index, neg_edge_index)
    
    loss = F.binary_cross_entropy_with_logits(scores, labels)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch+1:03d}, Loss: {loss.item():.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    z = model(data.x, data.train_pos_edge_index)
    pos_score = model.decode(z, data.test_pos_edge_index)
    neg_score = model.decode(z, data.test_neg_edge_index)
    
    scores = torch.cat([pos_score, neg_score], dim=0)
    labels = get_link_labels(data.test_pos_edge_index, data.test_neg_edge_index)
    
    auc_score = roc_auc_score(labels.cpu().numpy(), scores.cpu().numpy())
    print(f'Final Test AUC: {auc_score:.4f}')

# Visualization
embeddings_2d = TSNE(n_components=2).fit_transform(z.detach().numpy())
plt.figure(figsize=(8, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=10, c='blue', alpha=0.5)
plt.title("Node Embeddings Visualization")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.show()
