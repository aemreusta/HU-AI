import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary


# Load RNA sequences
def load_rna_data(file_path):
    with open(file_path, "r") as file:
        data = file.read().split(">")
    rna_sequences = {}
    for entry in data[1:]:
        lines = entry.strip().split("\n")
        header = lines[0].split()[0].split(".")[0]
        sequence = "".join(lines[1:])
        rna_sequences[header] = sequence
    print(f"Loaded {len(rna_sequences)} RNA sequences")
    return rna_sequences


# Load HGNC data
def load_hgnc_data(file_path):
    dtype = {
        "HGNC ID": str,
        "Approved symbol": str,
        "Approved name": str,
        "Chromosome location": str,
        "Chromosome": str,
        "Locus group": str,
        "Locus type": str,
        "HGNC family ID": str,
        "HGNC family name": str,
        "RefSeq accession": str,
        "NCBI gene ID": str,
        "Ensembl gene ID": str,
    }
    df = pd.read_csv(file_path, delimiter="\t", dtype=dtype)
    print("Loaded HGNC data with shape:", df.shape)
    return df


# Merge RNA sequences with HGNC data
def merge_data(df, rna_sequences):
    df["Sequence"] = df["RefSeq accession"].map(rna_sequences)
    missing_sequences = df["Sequence"].isnull().sum()
    df = df.dropna(subset=["Sequence"]).copy()
    print(f"Merged data contains {df.shape[0]} sequences")
    print(f"Missing sequences: {missing_sequences}")
    return df


# Clean RNA sequences
def clean_gene_string(sequence):
    valid_chars = {"A", "T", "C", "G"}
    return "".join([char if char in valid_chars else "X" for char in sequence])


# One-hot encode RNA sequences
def one_hot_encode_sequence(sequence):
    mapping = {
        "A": [1, 0, 0, 0, 0],
        "T": [0, 1, 0, 0, 0],
        "C": [0, 0, 1, 0, 0],
        "G": [0, 0, 0, 1, 0],
        "X": [0, 0, 0, 0, 1],
    }
    return [mapping[char] for char in sequence]


# Pad sequences to the same length
def pad_sequences(sequences, maxlen=1000):
    padded_sequences = np.zeros((len(sequences), maxlen, 5), dtype=int)
    for i, seq in enumerate(sequences):
        length = min(len(seq), maxlen)
        padded_sequences[i, :length, :] = seq[:length]
    return padded_sequences


# Preprocess data
def preprocess_data(hgnc_file_path, rna_file_path, maxlen=1000):
    # Load data
    hgnc_data = load_hgnc_data(hgnc_file_path)
    rna_sequences = load_rna_data(rna_file_path)

    # Merge data
    merged_data = merge_data(hgnc_data, rna_sequences)

    # Clean sequences
    merged_data["Sequence"] = merged_data["Sequence"].apply(clean_gene_string)

    # Limit the size to avoid memory issues
    merged_data = merged_data.iloc[:20000]

    # One-hot encode sequences
    one_hot_encoded_sequences = [
        one_hot_encode_sequence(seq) for seq in merged_data["Sequence"]
    ]

    # Pad sequences
    X = pad_sequences(one_hot_encoded_sequences, maxlen)

    # Encode labels
    y = pd.get_dummies(merged_data["Locus group"]).values

    print(
        f"Preprocessed data into {X.shape[0]} samples with {X.shape[1]} time steps each"
    )
    return np.array(X), np.array(y)


# Define function to calculate accuracy
def calculate_accuracy(y_true, y_pred):
    y_true = y_true.argmax(dim=1)
    y_pred = y_pred.argmax(dim=1)
    return accuracy_score(y_true.cpu(), y_pred.cpu())


# Define function to calculate F1 score
def calculate_f1(y_true, y_pred):
    y_true = y_true.argmax(dim=1)
    y_pred = y_pred.argmax(dim=1)
    return f1_score(y_true.cpu(), y_pred.cpu(), average="weighted")


# Define function to calculate MCC
def calculate_mcc(y_true, y_pred):
    y_true = y_true.argmax(dim=1)
    y_pred = y_pred.argmax(dim=1)
    return matthews_corrcoef(y_true.cpu(), y_pred.cpu())


# Define the RNN model in PyTorch
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out


# Training the model with cross-validation
def train_model_cv(X, y, k_folds=5):
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_results = []

    input_size = X.shape[2]  # One-hot encoding size
    hidden_size = 50
    output_size = y.shape[1]  # Number of classes
    num_epochs = 10
    learning_rate = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        print(f"Fold {fold + 1}")

        model = RNNModel(input_size, hidden_size, output_size).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        if fold == 0:
            print(summary(model, (X.shape[1], X.shape[2])))

        print(
            f"Train indices length: {len(train_idx)}, Validation indices length: {len(val_idx)}"
        )
        print(
            f"Max train index: {max(train_idx)}, Max validation index: {max(val_idx)}"
        )

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

        train_loader = DataLoader(
            RNADataset(X, y),  # Pass the full dataset and use subsampler for indices
            batch_size=32,
            sampler=train_subsampler,
        )
        val_loader = DataLoader(
            RNADataset(X, y),  # Pass the full dataset and use subsampler for indices
            batch_size=32,
            sampler=val_subsampler,
        )

        fold_history = {
            "fold": fold + 1,
            "train_time": None,
            "best_epoch": None,
            "train_loss": None,
            "train_acc": None,
            "train_f1": None,
            "train_mcc": None,
            "val_loss": None,
            "val_acc": None,
            "val_f1": None,
            "val_mcc": None,
        }

        history = {
            "fold": fold + 1,
            "train_loss": [],
            "train_acc": [],
            "train_f1": [],
            "train_mcc": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": [],
            "val_mcc": [],
        }

        start_time = time.time()
        best_epoch_val_f1 = 0

        for epoch in range(num_epochs):
            model.train()
            train_losses = []
            train_accuracies = []
            train_f1_scores = []
            train_mcc_scores = []

            for sequences, labels in train_loader:
                sequences = sequences.float().to(device)
                labels = labels.float().to(device)

                outputs = model(sequences)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())
                train_acc = calculate_accuracy(labels, outputs)
                train_f1 = calculate_f1(labels, outputs)
                train_mcc = calculate_mcc(labels, outputs)
                train_accuracies.append(train_acc)
                train_f1_scores.append(train_f1)
                train_mcc_scores.append(train_mcc)

            avg_train_loss = np.mean(train_losses)
            avg_train_acc = np.mean(train_accuracies)
            avg_train_f1 = np.mean(train_f1_scores)
            avg_train_mcc = np.mean(train_mcc_scores)

            model.eval()
            val_losses = []
            val_accuracies = []
            val_f1_scores = []
            val_mcc_scores = []

            with torch.no_grad():
                for sequences, labels in val_loader:
                    sequences = sequences.float().to(device)
                    labels = labels.float().to(device)

                    outputs = model(sequences)
                    loss = criterion(outputs, labels)

                    val_losses.append(loss.item())
                    val_acc = calculate_accuracy(labels, outputs)
                    val_f1 = calculate_f1(labels, outputs)
                    val_mcc = calculate_mcc(labels, outputs)
                    val_accuracies.append(val_acc)
                    val_f1_scores.append(val_f1)
                    val_mcc_scores.append(val_mcc)

            avg_val_loss = np.mean(val_losses)
            avg_val_acc = np.mean(val_accuracies)
            avg_val_f1 = np.mean(val_f1_scores)
            avg_val_mcc = np.mean(val_mcc_scores)

            history["train_loss"].append(avg_train_loss)
            history["train_acc"].append(avg_train_acc)
            history["train_f1"].append(avg_train_f1)
            history["train_mcc"].append(avg_train_mcc)
            history["val_loss"].append(avg_val_loss)
            history["val_acc"].append(avg_val_acc)
            history["val_f1"].append(avg_val_f1)
            history["val_mcc"].append(avg_val_mcc)

            print(
                f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, Train F1: {avg_train_f1:.4f}, Train MCC: {avg_train_mcc:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}, Val F1: {avg_val_f1:.4f}, Val MCC: {avg_val_mcc:.4f}"
            )

            if avg_val_f1 > best_epoch_val_f1:
                best_epoch_val_f1 = avg_val_f1
                fold_history["best_epoch"] = epoch + 1
                fold_history["train_loss"] = avg_train_loss
                fold_history["train_acc"] = avg_train_acc
                fold_history["train_f1"] = avg_train_f1
                fold_history["train_mcc"] = avg_train_mcc
                fold_history["val_loss"] = avg_val_loss
                fold_history["val_acc"] = avg_val_acc
                fold_history["val_f1"] = avg_val_f1
                fold_history["val_mcc"] = avg_val_mcc

                torch.save(model.state_dict(), f"model_fold_{fold + 1}.pt")

        end_time = time.time()
        hours, rem = divmod(end_time - start_time, 3600)
        minutes, seconds = divmod(rem, 60)
        fold_history["train_time"] = "{:0>2}:{:0>2}:{:05.2f}".format(
            int(hours), int(minutes), seconds
        )

        print(f"Fold {fold + 1} training completed in {fold_history['train_time']}")
        plot_training_history(history)
        fold_results.append(fold_history)

    return fold_results


# Visualization function
def plot_training_history(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    # Plot training & validation loss values
    plt.figure(figsize=(20, 10))

    # Subplot for training & validation loss values
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Subplot for training & validation accuracy values
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    # Subplot for training & validation F1-Score values
    plt.subplot(2, 2, 3)
    plt.plot(epochs, history["train_f1"], label="Train F1-Score")
    plt.plot(epochs, history["val_f1"], label="Validation F1-Score")
    plt.title("Training and Validation F1-Score")
    plt.xlabel("Epochs")
    plt.ylabel("F1-Score")
    plt.legend()

    # Subplot for training & validation MCC values
    plt.subplot(2, 2, 4)
    plt.plot(epochs, history["train_mcc"], label="Train MCC")
    plt.plot(epochs, history["val_mcc"], label="Validation MCC")
    plt.title("Training and Validation MCC")
    plt.xlabel("Epochs")
    plt.ylabel("MCC")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"training_history_fold_{history['fold']}.png")


# Adjust RNADataset to avoid out-of-bounds errors
class RNADataset(Dataset):
    def __init__(self, sequences, labels):
        assert len(sequences) == len(labels), "Mismatch between sequences and labels"
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# Visualize the dataset
def visualize_data(df):
    plt.figure(figsize=(12, 6))
    sns.countplot(
        data=df, x="Locus group", order=df["Locus group"].value_counts().index
    )
    plt.title("Distribution of Locus Groups")
    plt.xlabel("Locus Group")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    # plt.show()
    plt.savefig("locus_groups.png")

    df["Sequence Length"] = df["Sequence"].apply(len)

    plt.figure(figsize=(12, 6))
    sns.histplot(df["Sequence Length"], bins=50, kde=True)
    plt.title("Distribution of Sequence Lengths")
    plt.xlabel("Sequence Length")
    plt.ylabel("Frequency")
    # plt.show()
    plt.savefig("sequence_lengths.png")


def main():
    # File paths
    hgnc_file_path = "HGNC_results.txt"
    rna_file_path = "GRCh38_latest_rna.fna"

    # Preprocess data
    X, y = preprocess_data(hgnc_file_path, rna_file_path, maxlen=1000)

    print("Data preprocessing complete.")
    print("Shape of X:", X.shape)
    print("Shape of y:", y.shape)

    fold_results = train_model_cv(X, y)

    results_df = pd.DataFrame(fold_results)
    results_df.to_csv("fold_results.csv", index=False)

    # Load the best model based on validation F1 score
    model = RNNModel(X.shape[2], 50, y.shape[1])
    best_fold = max(fold_results, key=lambda x: x["val_f1"])["fold"]
    model.load_state_dict(torch.load(f"model_fold_{best_fold}.pt"))

    model.eval()

    # Generate predictions for the dataset
    predictions = []
    with torch.no_grad():
        for seq in X:
            seq_tensor = torch.tensor(seq).unsqueeze(0).float()
            output = model(seq_tensor)
            pred = output.argmax(dim=1).item()
            predictions.append(pred)

    # Create a DataFrame for the predictions
    hgnc_data = load_hgnc_data(hgnc_file_path)
    hgnc_data = hgnc_data.iloc[
        : len(predictions)
    ]  # Ensure it matches the length of predictions

    column_names = pd.get_dummies(
        hgnc_data["Locus group"]
    ).columns  # Get original column names
    hgnc_data["Prediction"] = [column_names[p] for p in predictions]

    # Save predictions to a file
    hgnc_data[["HGNC ID", "Prediction"]].to_csv(
        "HGNC_outputs.txt", index=False, header=False
    )


if __name__ == "__main__":
    main()
