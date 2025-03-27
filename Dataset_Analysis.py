import logomaker
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy.stats import entropy

# Function to compute PWM and information content
def calculate_pwm(sequences, center_length=30):
    seq_length = len(sequences[0])  
    start_idx = (seq_length - center_length) // 2  
    end_idx = start_idx + center_length  

    counts = {nuc: np.zeros(center_length) for nuc in "ATCG"}

    for seq in sequences:
        center_seq = seq[start_idx:end_idx]  
        for i, nuc in enumerate(center_seq):
            if nuc in counts:
                counts[nuc][i] += 1

    total_counts = sum(counts.values())
    pwm = {nuc: counts[nuc] / total_counts for nuc in counts}

    entropy_values = [entropy([counts[nuc][i] / total_counts[i] for nuc in "ATCG" if total_counts[i] > 0]) 
                      for i in range(center_length)]
    info_content = np.log2(4) - np.array(entropy_values)  

    return pd.DataFrame(pwm), info_content

# Function to plot and save sequence logos
def plot_pwm_logo(file_path, dataset_name, center_length=30):
    sequences, labels = [], []
    with open(file_path, 'r') as file:
        for line in file:
            seq, label = line.strip().split("\t")
            sequences.append(seq)
            labels.append(int(label))

    enhancer_sequences = [seq for seq, label in zip(sequences, labels) if label == 1]
    non_enhancer_sequences = [seq for seq, label in zip(sequences, labels) if label == 0]

    enhancer_pwm, enhancer_info = calculate_pwm(enhancer_sequences, center_length)
    non_enhancer_pwm, non_enhancer_info = calculate_pwm(non_enhancer_sequences, center_length)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    logomaker.Logo(enhancer_pwm, ax=axes[0], color_scheme='classic')
    axes[0].set_title(f"Enhancer Sequence Logo ({dataset_name})", fontsize=14, fontweight='bold')

    logomaker.Logo(non_enhancer_pwm, ax=axes[1], color_scheme='classic')
    axes[1].set_title(f"Non-Enhancer Sequence Logo ({dataset_name})", fontsize=14, fontweight='bold')

    axes[2].plot(enhancer_info, label="Enhancer Information Content", color="blue", lw=2)
    axes[2].plot(non_enhancer_info, label="Non-Enhancer Information Content", color="red", lw=2)
    axes[2].set_title("Shannon Information Content per Position", fontsize=14, fontweight='bold')
    axes[2].legend()

    plt.xlabel("Position in Sequence (Centered)", fontsize=12)
    plt.tight_layout()
    
    filename = f"sequence_logo_{dataset_name.replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300)
    print(f"Saved: {filename}")

# Function to plot and save nucleotide composition
def plot_nucleotide_distribution(file_path, dataset_name):
    sequences, labels = [], []
    with open(file_path, 'r') as file:
        for line in file:
            seq, label = line.strip().split("\t")
            sequences.append(seq)
            labels.append(int(label))

    enhancer_sequences = [seq for seq, label in zip(sequences, labels) if label == 1]
    non_enhancer_sequences = [seq for seq, label in zip(sequences, labels) if label == 0]

    def compute_composition(seq_list):
        total_length = sum(len(seq) for seq in seq_list)
        counts = Counter("".join(seq_list))
        return {nuc: counts[nuc] / total_length for nuc in "ATCG"}

    enhancer_comp = compute_composition(enhancer_sequences)
    non_enhancer_comp = compute_composition(non_enhancer_sequences)

    df = pd.DataFrame([enhancer_comp, non_enhancer_comp], index=["Enhancers", "Non-Enhancers"])
    df.plot(kind='bar', stacked=True, colormap="coolwarm", figsize=(8, 5))

    plt.title(f"Nucleotide Composition ({dataset_name})", fontsize=14, fontweight='bold')
    plt.ylabel("Proportion", fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(title="Nucleotide")
    
    filename = f"nucleotide_composition_{dataset_name.replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300)
    print(f"Saved: {filename}")

# Function to plot and save k-mer distribution
def plot_kmer_distribution(file_path, dataset_name, k=3):
    sequences, labels = [], []
    with open(file_path, 'r') as file:
        for line in file:
            seq, label = line.strip().split("\t")
            sequences.append(seq)
            labels.append(int(label))

    enhancer_sequences = [seq for seq, label in zip(sequences, labels) if label == 1]
    non_enhancer_sequences = [seq for seq, label in zip(sequences, labels) if label == 0]

    def compute_kmer_freq(seq_list, k):
        kmers = [seq[i:i+k] for seq in seq_list for i in range(len(seq) - k + 1)]
        return dict(Counter(kmers).most_common(10))

    enhancer_kmers = compute_kmer_freq(enhancer_sequences, k)
    non_enhancer_kmers = compute_kmer_freq(non_enhancer_sequences, k)

    df = pd.DataFrame([enhancer_kmers, non_enhancer_kmers], index=["Enhancers", "Non-Enhancers"]).T
    df.plot(kind='bar', colormap="viridis", figsize=(10, 5))

    plt.title(f"Top {k}-mer Frequency ({dataset_name})", fontsize=14, fontweight='bold')
    plt.ylabel("Frequency", fontsize=12)
    plt.xlabel(f"{k}-mer Sequences", fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title="Category")
    
    filename = f"kmer_distribution_{dataset_name.replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300)
    print(f"Saved: {filename}")

# Generate and save all figures
datasets = [("Dataset/Enhancer/enhancer_train_data.txt", "Training"), 
            ("Dataset/Enhancer/enhancer_independent_data.txt", "Test")]

for file_path, dataset_name in datasets:
    plot_pwm_logo(file_path, dataset_name, center_length=30)
    plot_nucleotide_distribution(file_path, dataset_name)
    plot_kmer_distribution(file_path, dataset_name, k=3)
