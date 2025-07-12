# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 16:56:53 2025

@author: Rehman   
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from gensim.models import Word2Vec
from torch.utils.data import Dataset, DataLoader
import random
import os
# Set random seed
def set_seed(seed=43):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
set_seed(43)


def set_seed(seed=43):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(43)  


class DNAEncoding:
    @staticmethod
    def one_hot_encode(sequence):
        mapping = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'C': [0, 0, 1, 0], 'G': [0, 0, 0, 1]}
        return torch.tensor([mapping[base] for base in sequence if base in mapping], dtype=torch.float).T
    
    @staticmethod
    def kmer_encoding(sequence, k=3):
        k_mers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
        return k_mers
    
    @staticmethod
    def physicochemical_encoding(sequence):
        pc_map = {'A': [1.0, 0.8, 0.5], 'T': [0.7, 0.5, 0.2], 'C': [0.6, 0.4, 0.3], 'G': [0.9, 0.6, 0.7]}
        return torch.tensor([pc_map[base] for base in sequence if base in pc_map], dtype=torch.float).T
    
    @staticmethod
    def train_word2vec_kmers(sequences, k=3, vector_size=100):
        kmer_sequences = [DNAEncoding.kmer_encoding(seq, k) for seq in sequences]
        model = Word2Vec(kmer_sequences, vector_size=vector_size, window=5, min_count=1, workers=4)
        return model
    
    @staticmethod
    def encode_with_word2vec(sequence, model, k=3):
        k_mers = DNAEncoding.kmer_encoding(sequence, k)
        encoded = [model.wv[k_mer] for k_mer in k_mers if k_mer in model.wv]
        return torch.tensor(encoded, dtype=torch.float).T if encoded else torch.zeros((model.vector_size, len(k_mers)))


class DNASequenceDataset(Dataset):
    def __init__(self, file_path, encoding_type="one_hot", w2v_model=None):
        self.sequences, self.labels = self.load_data(file_path)
        self.encoding_type = encoding_type
        self.w2v_model = w2v_model
    
    def load_data(self, file_path):
        sequences, labels = [], []
        with open(file_path, 'r') as file:
            for line in file:
                seq, label = line.strip().split("\t")
                sequences.append(seq)
                labels.append(int(label))
        return sequences, labels
    
    def encode_sequence(self, sequence):
        if self.encoding_type == "one_hot":
            return DNAEncoding.one_hot_encode(sequence)
        elif self.encoding_type == "physicochemical":
            return DNAEncoding.physicochemical_encoding(sequence)
        elif self.encoding_type == "word2vec" and self.w2v_model:
            return DNAEncoding.encode_with_word2vec(sequence, self.w2v_model)
        else:
            return DNAEncoding.one_hot_encode(sequence)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        encoded_seq = self.encode_sequence(self.sequences[idx])
        return encoded_seq, torch.tensor(self.labels[idx], dtype=torch.long)




# Function to plot and save t-SNE visualization
def plot_tsne(features, labels, title, epoch):
    # Create 'tsne-folder' if it doesn't exist
    os.makedirs('tsne-folder', exist_ok=True)
    
    # Map labels (0, 1) to class names ("non-enhancer", "enhancer")
    label_names = ["non-enhancer", "enhancer"]
    mapped_labels = [label_names[label] for label in labels]
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=43)
    reduced_features = tsne.fit_transform(features)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='coolwarm', alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title(f"{title} - Epoch {epoch}")
    
    # Save the plot to the 'tsne-folder' directory
    plt.savefig(f"tsne-folder/tsne_{title.lower().replace(' ', '_')}_epoch_{epoch}.png")
    plt.close()

def extract_features(model, loader, device, stage):
    model.eval()
    all_features, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Extract features based on stage
            if stage == "input":
                features = inputs.reshape(inputs.size(0), -1).cpu().numpy()
            else:
                # Forward pass through convolutional layers and attention
                x = model.conv1(inputs)
                if stage == "conv1":
                    features = x.reshape(x.size(0), -1).cpu().numpy()
                else:
                    x = model.relu(model.batchnorm1(x))
                    x = model.maxpool(x)
                    
                    x = model.conv2(x)
                    if stage == "conv2":
                        features = x.reshape(x.size(0), -1).cpu().numpy()
                    else:
                        x = model.relu(model.batchnorm2(x))
                        x = model.maxpool(x)
                        
                        x = model.conv3(x)
                        if stage == "conv3":
                            features = x.reshape(x.size(0), -1).cpu().numpy()
                        else:
                            x = model.relu(model.batchnorm3(x))
                            x = model.maxpool(x)
                            
                            # Apply attention after conv3
                            x = x.permute(0, 2, 1)  # Change shape for attention (batch, seq_len, features)
                            x, _ = model.attn_layer(x, x, x)
                            x = x.permute(0, 2, 1)  # Revert shape back (batch, features, seq_len)
                            
                            if stage == "attention":
                                features = x.reshape(x.size(0), -1).cpu().numpy()
                            else:
                                x = x.reshape(x.size(0), -1)
                                features = x.cpu().numpy()

            all_features.append(features)
            all_labels.append(labels.cpu().numpy())
    
    return np.vstack(all_features), np.hstack(all_labels)



# Define CNN model class with feature extraction at each stage
class CNNEnhancerPredictor(nn.Module):
    def __init__(self, input_channels=4, output_size=2, sequence_length=100):
        super(CNNEnhancerPredictor, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=2)
        )
        self.attn_layer = nn.MultiheadAttention(embed_dim=64, num_heads=8, batch_first=True)
        self.fc_layers = nn.Sequential(
            nn.Linear(1536, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, output_size)
        )
    
    def forward(self, x):
        # Apply convolution layers
        x = self.conv_layers(x)
        
        # Change the dimensions for attention layer
        x = x.permute(0, 2, 1)
        
        # Pass through attention layer and return attention weights
        attn_output, attn_weights = self.attn_layer(x, x, x)
        
        # Reverse permute the attention output back to original shape
        x = attn_output.permute(0, 2, 1)
        
        # Flatten and pass through fully connected layers
        x = x.reshape(x.size(0), -1)
        x = self.fc_layers(x)
        
        return x, attn_weights  # Return both the output and attention weights


# Load dataset
train_sequences = [line.strip().split("\t")[0] for line in open('Dataset/Enhancer/enhancer_train_data.txt')]
w2v_model = Word2Vec.load("word2vec_model.bin")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoding_type = "word2vec"  
train_dataset = DNASequenceDataset('Dataset/Enhancer/enhancer_train_data.txt', encoding_type, w2v_model)
test_dataset = DNASequenceDataset('Dataset/Enhancer/enhancer_independent_data.txt', encoding_type, w2v_model)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

model = CNNEnhancerPredictor(input_channels=w2v_model.vector_size if encoding_type == "word2vec" else 4).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

import seaborn as sns
import matplotlib.pyplot as plt

# Visualize attention weights as heatmap after averaging over attention heads
def visualize_attention_weights(attn_weights, epoch, stage):
    # Average the attention weights across all attention heads
    avg_attn_weights = attn_weights.mean(dim=1)  # Shape: (batch_size, sequence_length)

    # For visualization, we'll choose a specific batch (e.g., the first batch)
    avg_attn_weights = avg_attn_weights[0]  # Assuming batch_size > 0

    # Ensure the shape is 2D (sequence_length,) by removing any singleton dimensions
    avg_attn_weights = avg_attn_weights.squeeze()  # Remove any extra dimensions
    
    # Convert to numpy array for visualization
    avg_attn_weights = avg_attn_weights.cpu().detach().numpy()

    # Create a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(avg_attn_weights[None, :], cmap='viridis', cbar=True, annot=False, xticklabels=False, yticklabels=False)
    plt.title(f"Attention Weights ({stage}) - Epoch {epoch}")
    plt.xlabel('Sequence Position')
    plt.ylabel('Query Position')
    plt.savefig(f"tsne-folder/attention_weights_{stage}_epoch_{epoch}.png")
    plt.close()

# In the training loop, call visualize_attention_weights for each epoch
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass (outputs and attention weights)
        outputs, attn_weights = model(inputs)
        
        # Loss calculation
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader)}')
    
    # Visualize attention weights for each epoch and stage
    visualize_attention_weights(attn_weights, epoch, stage="Attention")
