import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_curve, roc_auc_score
import numpy as np
from gensim.models import Word2Vec
from itertools import product
import random
 

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


import torch
import torch.nn as nn
import torch.nn.functional as F



class CNNEnhancerPredictor(nn.Module):
    def __init__(self, input_channels=4, output_size=2, sequence_length=100):
        super(CNNEnhancerPredictor, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, 256, kernel_size=3, padding=1),  # Adjusted padding
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

        # Dynamically calculate the output size after convolutions
        with torch.no_grad():
            sample_input = torch.randn(1, input_channels, sequence_length)
            conv_output = self.conv_layers(sample_input)
            self.flatten_dim = conv_output.shape[1] * conv_output.shape[2]

        # Attention Layer (Ensuring proper embed_dim)
        self.attn_layer = nn.MultiheadAttention(embed_dim=64, num_heads=8, batch_first=True)

        # Fully Connected Layers
        self.fc_layers = nn.Sequential(
            nn.Linear(1536, 256),
            nn.ReLU(),
            nn.Dropout(0.4),  # Reduced dropout
            nn.Linear(256, output_size)
        )

    def forward(self, x):
        x = self.conv_layers(x)  # (batch, 64, new_seq_len)

        # Attention Mechanism (Making sure the tensor shape matches the required input for MultiHeadAttention)
        x = x.permute(0, 2, 1)  # Convert to (batch, seq_len, channels) for attention
        x, _ = self.attn_layer(x, x, x)
        x = x.permute(0, 2, 1)  # Convert back to (batch, channels, seq_len)

        x = x.reshape(x.size(0), -1)  # Flatten dynamically
        x = self.fc_layers(x)
        return x



train_sequences = [line.strip().split("\t")[0] for line in open('Dataset/Enhancer/enhancer_train_data.txt')]



#w2v_model = DNAEncoding.train_word2vec_kmers(train_sequences, k=3)
#w2v_model.save("word2vec_model.bin")  # Save model

# Next time, load the saved model instead of retraining
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

num_epochs = 72  
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader)}')

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, matthews_corrcoef, recall_score, precision_score, confusion_matrix

torch.save(model.state_dict(), 'enhancer_model_weights.pth')
print("Model weights saved successfully.")



model.eval()
all_preds, all_labels, all_probs = [], [], []
softmax = nn.Softmax(dim=1)

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        probs = softmax(outputs)
        _, preds = torch.max(probs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy()[:, 1])

accuracy = accuracy_score(all_labels, all_preds)
roc_auc = roc_auc_score(all_labels, all_probs)
f1 = f1_score(all_labels, all_preds)
mcc = matthews_corrcoef(all_labels, all_preds)

tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()

sensitivity = recall_score(all_labels, all_preds)
specificity = tn / (tn + fp)

precision = precision_score(all_labels, all_preds)

print(f'Accuracy: {accuracy:.4f}')
print(f'ROC-AUC: {roc_auc:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'MCC: {mcc:.4f}')
print(f'Sensitivity: {sensitivity:.4f}')
print(f'Specificity: {specificity:.4f}')
print(f'Precision: {precision:.4f}')
