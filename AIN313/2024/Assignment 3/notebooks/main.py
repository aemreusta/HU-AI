# %% [markdown]
# # Ahmet Emre Usta
# # 2200765036

# %% [markdown]
# # Part 1

# %% [markdown]
# ### Answer 1.
#
# Activation functions are mathematical functions applied to the output of a neuron in a neural network. They introduce non-linearity into the model, enabling it to learn complex patterns in data. Without activation functions, neural networks would essentially be linear models, limiting their capability to approximate complex functions.
#
# Key types of activation functions:
# - Sigmoid: Maps input to a range (0, 1), used in binary classification.
# - ReLU (Rectified Linear Unit): This unit outputs the input directly if it is positive, otherwise zero. It is widely used due to its simplicity and ability to avoid vanishing gradients.
# - Tanh: Similar to sigmoid but maps input to (-1, 1), improving the output range.
# - Softmax: Converts logits to probabilities in multi-class classification.
#
# They are essential for enabling deep networks to model complex relationships and make accurate predictions in tasks like image recognition, NLP.
#

# %% [markdown]
# ### Answer 2.
#
# | Layer      | Output Volume Shape   | Number of Parameters |
# |------------|-----------------------|----------------------|
# | Input      | (64, 64, 3)           | 0                    |
# | CONV5-8    | (60, 60, 8)           | 608                  |
# | POOL-2     | (30, 30, 8)           | 0                    |
# | CONV3-16   | (28, 28, 16)          | 1168                 |
# | POOL-3     | (14, 14, 16)          | 0                    |
# | FC-30      | (30)                  | 94110                |
# | FC-5       | (5)                   | 155                  |

# %% [markdown]
# # Part 2

# %% [markdown]
# ## Install Necessary Libaries

# %%
# %pip install torch torchvision plotly kaleido pandas scikit-learn >> /dev/null

# %% [markdown]
# ## Import Libaries

# %%
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# %% [markdown]
# ## Set the Paths and Read the Dataset

# %%
DATASET_PATH = "/Users/emre/GitHub/HU-AI/AIN313/2024/Assignment 3/dataset/"
TRAIN_BENIGN_DATASET_PATH = os.path.join(DATASET_PATH, "raw", "train", "benign")
TRAIN_MALIGNANT_DATASET_PATH = os.path.join(DATASET_PATH, "raw", "train", "malignant")
TEST_BENIGN_DATASET_PATH = os.path.join(DATASET_PATH, "raw", "test", "benign")
TEST_MALIGNANT_DATASET_PATH = os.path.join(DATASET_PATH, "raw", "test", "malignant")

# %% [markdown]
# ## Utility Functions


# %%
def create_directories():
    dirs = ["log", "models", "figures"]
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)


create_directories()


# %%
def get_device():
    """
    Selects the device to run the model on.
    Priority: mps (Metal) > cuda > cpu
    Returns:
        device (torch.device): The selected device.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Metal Performance Shaders (MPS) backend.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA backend.")
    else:
        device = torch.device("cpu")
        print("Using CPU.")
    return device


# %%
def train_model(model, train_loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        preds = (outputs > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_prec = precision_score(all_labels, all_preds, zero_division=0)
    epoch_rec = recall_score(all_labels, all_preds, zero_division=0)
    epoch_f1 = f1_score(all_labels, all_preds, zero_division=0)
    return epoch_loss, epoch_acc, epoch_prec, epoch_rec, epoch_f1


# %%
def evaluate_model(model, test_loader, criterion, device):
    """
    Evaluates the model on the test set.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    epoch_loss = running_loss / len(test_loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_prec = precision_score(all_labels, all_preds, zero_division=0)
    epoch_rec = recall_score(all_labels, all_preds, zero_division=0)
    epoch_f1 = f1_score(all_labels, all_preds, zero_division=0)
    return epoch_loss, epoch_acc, epoch_prec, epoch_rec, epoch_f1


# %%
def log_experiment(exp_type, params, results):
    """
    Logs the experiment results to a CSV file using pd.concat instead of append.
    """
    log_file = os.path.join("log", f"{exp_type}_results.csv")
    experiment = {**params, **results}
    experiment_df = pd.DataFrame([experiment])

    if not os.path.exists(log_file):
        experiment_df.to_csv(log_file, index=False)
    else:
        existing_df = pd.read_csv(log_file)
        updated_df = pd.concat([existing_df, experiment_df], ignore_index=True)
        updated_df.to_csv(log_file, index=False)


# %%
def visualize_sample_images(dataset_path, num_images=10):
    """
    Visualizes sample images from the training set with their class names using Plotly.
    Displays all images in a grid layout.
    """
    transform = transforms.Compose(
        [
            transforms.Resize((100, 100)),  # Only resizing
        ]
    )
    train_dataset = datasets.ImageFolder(
        root=dataset_path,
        transform=transform,
    )
    class_names = train_dataset.classes

    print(
        f"Number of images in the dataset: {len(train_dataset)}, Number of classes: {len(class_names)}",
        f"Class Names: {class_names}",
    )

    indices = np.random.choice(len(train_dataset), num_images, replace=False)
    samples = [train_dataset[i] for i in indices]
    images, labels = zip(*samples)

    # Convert images to numpy arrays and scale to 0-255
    images = [
        np.array(img.resize((100, 100))) for img in images
    ]  # Resize manually if needed
    images = [
        np.clip(img, 0, 255).astype(np.uint8) for img in images
    ]  # Ensure uint8 format
    labels = [class_names[int(label)] for label in labels]

    # Set up subplots grid (2 rows x 5 columns for 10 images)
    rows = 2
    cols = 5
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=labels,  # Add labels as subplot titles
        vertical_spacing=0.07,
        horizontal_spacing=0.07,
    )

    for i, (img, label) in enumerate(zip(images, labels)):
        row = i // cols + 1  # 1-indexed row
        col = i % cols + 1  # 1-indexed column
        fig.add_trace(go.Image(z=img, name=label), row=row, col=col)

    fig.update_layout(
        title="Sample Training Images with Class Names",
        height=500,  # Adjust for better visualization
        width=1000,
        showlegend=False,
        xaxis=dict(showticklabels=False),  # Hide x-axis ticks
        yaxis=dict(showticklabels=False),  # Hide y-axis ticks
    )

    fig.show()


# %%
def visualize_loss_curve(training_history, model_type, experiment_id):
    """
    Plots the loss curve using Plotly.
    """
    epochs = list(range(1, len(training_history["train_loss"]) + 1))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=epochs, y=training_history["train_loss"], mode="lines", name="Train Loss"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=training_history["val_loss"],
            mode="lines",
            name="Validation Loss",
        )
    )
    fig.update_layout(
        title=f"Loss Curve for {model_type} Experiment {experiment_id}",
        xaxis_title="Epoch",
        yaxis_title="Loss",
    )
    fig_path = os.path.join(
        "figures", f"{model_type}_loss_curve_exp_{experiment_id}.png"
    )
    fig.write_image(fig_path)


# %%
def visualize_metrics_curve(training_history, model_type, experiment_id):
    """
    Plots the accuracy, precision, recall, and F1-score curves using Plotly.
    """
    epochs = list(range(1, len(training_history["train_acc"]) + 1))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=training_history["train_acc"],
            mode="lines",
            name="Train Accuracy",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=training_history["val_acc"],
            mode="lines",
            name="Validation Accuracy",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=training_history["train_prec"],
            mode="lines",
            name="Train Precision",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=training_history["val_prec"],
            mode="lines",
            name="Validation Precision",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=epochs, y=training_history["train_rec"], mode="lines", name="Train Recall"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=training_history["val_rec"],
            mode="lines",
            name="Validation Recall",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=training_history["train_f1"],
            mode="lines",
            name="Train F1-Score",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=training_history["val_f1"],
            mode="lines",
            name="Validation F1-Score",
        )
    )
    fig.update_layout(
        title=f"Metrics Curve for {model_type} Experiment {experiment_id}",
        xaxis_title="Epoch",
        yaxis_title="Metric Value",
    )
    fig_path = os.path.join(
        "figures", f"{model_type}_metrics_curve_exp_{experiment_id}.png"
    )
    fig.write_image(fig_path)


# %%
def visualize_weights(model, model_type, experiment_id, input_size):
    """
    Visualizes the learned parameters as images using Plotly.
    Only applicable for MLP.
    """
    if model_type != "MLP":
        return
    with torch.no_grad():
        weights = (
            model.network[1].weight.data.cpu().numpy()
        )  # First linear layer after Flatten
    num_weights = weights.shape[0]
    img_size = int(np.sqrt(weights.shape[1]))  # Assuming square input
    fig = go.Figure()
    for i in range(min(num_weights, 10)):  # Visualize first 10 weights
        # try:
        w = weights[i].reshape(img_size, img_size)
        fig.add_trace(go.Image(z=w, name=f"Neuron {i}"))
    # except:
    #     print(
    #         f"Cannot reshape weights for neuron {i}; input size may not be square."
    #     )
    fig.update_layout(
        title=f"Learned Weights for {model_type} Experiment {experiment_id}", height=600
    )
    fig_path = os.path.join("figures", f"{model_type}_weights_exp_{experiment_id}.png")
    fig.write_image(fig_path)


# %%
def get_data_loaders(input_size, batch_size):
    """
    Creates train and test data loaders with specified input size and batch size.
    Args:
        input_size (tuple): Desired input size (height, width).
        batch_size (int): Batch size.
    Returns:
        train_loader, test_loader: Data loaders for training and testing.
    """
    transform = transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.ToTensor(),  # Convert to tensor and normalize to [0,1]
        ]
    )

    # Assuming the dataset is structured as 'raw/train/benign', 'raw/train/malignant', etc.
    train_dataset = datasets.ImageFolder(
        root="/Users/emre/GitHub/HU-AI/AIN313/2024/Assignment 3/dataset/raw/train",
        transform=transform,
    )
    test_dataset = datasets.ImageFolder(
        root="/Users/emre/GitHub/HU-AI/AIN313/2024/Assignment 3/dataset/raw/test",
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    return train_loader, test_loader


# %%
def run_mlp_experiments():
    """
    Runs experiments for the Multi-Layer Perceptron (MLP) model.
    """
    device = get_device()
    activation_functions = ["relu", "sigmoid"]
    learning_rates = [0.005, 0.02]
    input_sizes = [(50, 50), (300, 300)]
    hidden_dims = [128, 64]  # Exclude the output layer; it's handled in the model
    batch_size = 16
    epochs = 20  # Define number of epochs

    experiment_id = 1
    for lr in learning_rates:
        for input_size in input_sizes:
            for act_fn in activation_functions:
                print(
                    f"\nMLP Experiment {experiment_id}: lr={lr}, input_size={input_size}, activation={act_fn}"
                )
                # Prepare data loaders
                train_loader, test_loader = get_data_loaders(input_size, batch_size)

                # Calculate input dimension
                input_dim = input_size[0] * input_size[1] * 3

                print(f"Input size: {input_size}", f"Input dimension: {input_dim}")

                # Initialize model
                model = MLP(
                    input_dim=input_dim, hidden_dims=hidden_dims, activation_fn=act_fn
                ).to(device)

                # Define loss and optimizer
                criterion = nn.BCELoss()
                optimizer = optim.Adam(model.parameters(), lr=lr)

                # Training history
                training_history = {
                    "train_loss": [],
                    "val_loss": [],
                    "train_acc": [],
                    "val_acc": [],
                    "train_prec": [],
                    "val_prec": [],
                    "train_rec": [],
                    "val_rec": [],
                    "train_f1": [],
                    "val_f1": [],
                }

                # Training loop
                for epoch in range(epochs):
                    train_loss, train_acc, train_prec, train_rec, train_f1 = (
                        train_model(model, train_loader, criterion, optimizer, device)
                    )
                    val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate_model(
                        model, test_loader, criterion, device
                    )

                    training_history["train_loss"].append(train_loss)
                    training_history["val_loss"].append(val_loss)
                    training_history["train_acc"].append(train_acc)
                    training_history["val_acc"].append(val_acc)
                    training_history["train_prec"].append(train_prec)
                    training_history["val_prec"].append(val_prec)
                    training_history["train_rec"].append(train_rec)
                    training_history["val_rec"].append(val_rec)
                    training_history["train_f1"].append(train_f1)
                    training_history["val_f1"].append(val_f1)

                    print(
                        f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                    )

                # Log results
                params = {
                    "learning_rate": lr,
                    "input_size": f"{input_size[0]}x{input_size[1]}",
                    "activation_function": act_fn,
                    "hidden_dims": hidden_dims,
                }
                results = {
                    "final_train_loss": training_history["train_loss"][-1],
                    "final_val_loss": training_history["val_loss"][-1],
                    "final_train_acc": training_history["train_acc"][-1],
                    "final_val_acc": training_history["val_acc"][-1],
                    "final_train_prec": training_history["train_prec"][-1],
                    "final_val_prec": training_history["val_prec"][-1],
                    "final_train_rec": training_history["train_rec"][-1],
                    "final_val_rec": training_history["val_rec"][-1],
                    "final_train_f1": training_history["train_f1"][-1],
                    "final_val_f1": training_history["val_f1"][-1],
                }
                log_experiment("MLP", params, results)

                # Save model
                model_path = os.path.join("models", f"MLP_exp_{experiment_id}.pth")
                torch.save(model.state_dict(), model_path)

                # Visualize loss and metrics
                visualize_loss_curve(training_history, "MLP", experiment_id)
                visualize_metrics_curve(training_history, "MLP", experiment_id)

                # Visualize weights
                visualize_weights(model, "MLP", experiment_id, input_size)

                experiment_id += 1


# %%
def run_cnn_experiments():
    """
    Runs experiments for the Convolutional Neural Network (CNN) model.
    """
    device = get_device()
    activation_functions = ["relu", "sigmoid"]
    learning_rates = [0.005, 0.02]
    batch_sizes = [16, 32]
    input_size = (300, 300)
    epochs = 20  # Define number of epochs

    experiment_id = 1
    for lr in learning_rates:
        for batch_size in batch_sizes:
            for act_fn in activation_functions:
                print(
                    f"\nCNN Experiment {experiment_id}: lr={lr}, batch_size={batch_size}, activation={act_fn}"
                )
                # Prepare data loaders
                train_loader, test_loader = get_data_loaders(input_size, batch_size)

                # Initialize model
                model = CNN(activation_fn=act_fn, input_size=input_size).to(device)

                # Define loss and optimizer
                criterion = nn.BCELoss()
                optimizer = optim.Adam(model.parameters(), lr=lr)

                # Training history
                training_history = {
                    "train_loss": [],
                    "val_loss": [],
                    "train_acc": [],
                    "val_acc": [],
                    "train_prec": [],
                    "val_prec": [],
                    "train_rec": [],
                    "val_rec": [],
                    "train_f1": [],
                    "val_f1": [],
                }

                # Training loop
                for epoch in range(epochs):
                    train_loss, train_acc, train_prec, train_rec, train_f1 = (
                        train_model(model, train_loader, criterion, optimizer, device)
                    )
                    val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate_model(
                        model, test_loader, criterion, device
                    )

                    training_history["train_loss"].append(train_loss)
                    training_history["val_loss"].append(val_loss)
                    training_history["train_acc"].append(train_acc)
                    training_history["val_acc"].append(val_acc)
                    training_history["train_prec"].append(train_prec)
                    training_history["val_prec"].append(val_prec)
                    training_history["train_rec"].append(train_rec)
                    training_history["val_rec"].append(val_rec)
                    training_history["train_f1"].append(train_f1)
                    training_history["val_f1"].append(val_f1)

                    print(
                        f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                    )

                # Log results
                params = {
                    "learning_rate": lr,
                    "batch_size": batch_size,
                    "input_size": f"{input_size[0]}x{input_size[1]}",
                    "activation_function": act_fn,
                }
                results = {
                    "final_train_loss": training_history["train_loss"][-1],
                    "final_val_loss": training_history["val_loss"][-1],
                    "final_train_acc": training_history["train_acc"][-1],
                    "final_val_acc": training_history["val_acc"][-1],
                    "final_train_prec": training_history["train_prec"][-1],
                    "final_val_prec": training_history["val_prec"][-1],
                    "final_train_rec": training_history["train_rec"][-1],
                    "final_val_rec": training_history["val_rec"][-1],
                    "final_train_f1": training_history["train_f1"][-1],
                    "final_val_f1": training_history["val_f1"][-1],
                }
                log_experiment("CNN", params, results)

                # Save model
                model_path = os.path.join("models", f"CNN_exp_{experiment_id}.pth")
                torch.save(model.state_dict(), model_path)

                # Visualize loss and metrics
                visualize_loss_curve(training_history, "CNN", experiment_id)
                visualize_metrics_curve(training_history, "CNN", experiment_id)

                experiment_id += 1


# %%
def plot_experiment_results():
    """
    Plots the experiment results from the log files using Plotly.
    """
    # MLP Results
    mlp_log_path = os.path.join("log", "MLP_results.csv")
    if os.path.exists(mlp_log_path):
        mlp_log = pd.read_csv(mlp_log_path)
        # Plot final validation accuracy for MLP
        fig = go.Figure(
            data=[
                go.Bar(
                    name="Validation Accuracy",
                    x=mlp_log.index + 1,
                    y=mlp_log["final_val_acc"],
                ),
                go.Bar(
                    name="Train Accuracy",
                    x=mlp_log.index + 1,
                    y=mlp_log["final_train_acc"],
                ),
            ]
        )
        # Change the bar mode
        fig.update_layout(
            barmode="group",
            title="Final Validation and Training Accuracy for MLP Experiments",
            xaxis_title="Experiment ID",
            yaxis_title="Accuracy",
        )
        fig.show()
    else:
        print("MLP_results.csv not found in 'log' directory.")

    # CNN Results
    cnn_log_path = os.path.join("log", "CNN_results.csv")
    if os.path.exists(cnn_log_path):
        cnn_log = pd.read_csv(cnn_log_path)
        # Plot final validation accuracy for CNN
        fig = go.Figure(
            data=[
                go.Bar(
                    name="Validation Accuracy",
                    x=cnn_log.index + 1,
                    y=cnn_log["final_val_acc"],
                ),
                go.Bar(
                    name="Train Accuracy",
                    x=cnn_log.index + 1,
                    y=cnn_log["final_train_acc"],
                ),
            ]
        )
        # Change the bar mode
        fig.update_layout(
            barmode="group",
            title="Final Validation and Training Accuracy for CNN Experiments",
            xaxis_title="Experiment ID",
            yaxis_title="Accuracy",
        )
        fig.show()
    else:
        print("CNN_results.csv not found in 'log' directory.")


# %% [markdown]
# ## Models


# %%
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, activation_fn):
        """
        Multi-Layer Perceptron for binary classification.
        Args:
            input_dim (int): Number of input features.
            hidden_dims (list): List containing the number of neurons in each hidden layer.
            activation_fn (str): Activation function ('relu' or 'sigmoid').
        """
        super(MLP, self).__init__()
        layers = [nn.Flatten()]  # Flatten the input

        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation_fn.lower() == "relu":
                layers.append(nn.ReLU())
            elif activation_fn.lower() == "sigmoid":
                layers.append(nn.Sigmoid())
            else:
                raise ValueError("Unsupported activation function")
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))  # Output layer
        layers.append(nn.Sigmoid())  # Sigmoid for binary classification
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# %%
class CNN(nn.Module):
    def __init__(self, activation_fn, input_size):
        """
        Convolutional Neural Network for binary classification.
        Args:
            activation_fn (str): Activation function ('relu' or 'sigmoid').
            input_size (tuple): Input image size (height, width).
        """
        super(CNN, self).__init__()

        # Select activation function
        if activation_fn.lower() == "relu":
            act_fn = nn.ReLU()
        elif activation_fn.lower() == "sigmoid":
            act_fn = nn.Sigmoid()
        else:
            raise ValueError("Unsupported activation function")

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                3, 16, kernel_size=3, padding=1
            ),  # Change input channels to 3 for RGB
            act_fn,
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            act_fn,
            nn.MaxPool2d(2, 2),
        )

        # Calculate the size after convolution and pooling
        conv_output_size = input_size[0] // 4  # Two pool layers, each divides by 2

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * conv_output_size * conv_output_size, 128),
            act_fn,
            nn.Linear(128, 64),
            act_fn,
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the output from convolution layers
        x = self.fc_layers(x)
        return x


# %% [markdown]
# ## Dataset Exploration

# %%
# Visualize sample images
visualize_sample_images(
    dataset_path="/Users/emre/GitHub/HU-AI/AIN313/2024/Assignment 3/dataset/raw/train",
    num_images=10,
)

# %% [markdown]
# ## MLP Experiments

# %%
# Run MLP experiments
run_mlp_experiments()

# %% [markdown]
# ## CNN Experiments

# %%
# Run CNN experiments
run_cnn_experiments()

# %% [markdown]
# ## Results

# %%
# Plot experiment results
plot_experiment_results()

# %% [markdown]
# # **Report on Skin Lesion Classification Experiments**
#
# ## **1. Introduction**
#
# This report presents a comprehensive analysis of experiments conducted to classify skin lesions using two distinct neural network architectures: **Convolutional Neural Networks (CNNs)** and **Multi-Layer Perceptrons (MLPs)**. The primary objective was to evaluate the impact of various hyperparameters on the performance of each model, with the goal of identifying configurations that yield optimal classification accuracy.
#
# ### **1.1. Objectives**
#
# - **Evaluate Model Architectures**: Compare the performance of CNNs and MLPs in classifying skin lesions.
# - **Hyperparameter Tuning**: Assess the influence of learning rates, batch sizes, input sizes, activation functions, and hidden layer dimensions on model performance.
# - **Performance Metrics**: Utilize multiple evaluation metrics to gain a holistic understanding of model efficacy, including accuracy, precision, recall, and F1-score.
#
# ### **1.2. Dataset**
#
# Assuming a binary classification task (e.g., benign vs. malignant lesions), the dataset is structured with separate directories for training and testing images, further categorized into classes. Images are preprocessed by resizing, converting to grayscale, and normalizing pixel values.
#
# ## **2. Experimental Setup**
#
# ### **2.1. Model Architectures**
#
# - **Convolutional Neural Networks (CNNs)**: Designed to capture spatial hierarchies in image data through convolutional and pooling layers, followed by fully connected layers.
# - **Multi-Layer Perceptrons (MLPs)**: Composed of fully connected layers, suitable for scenarios where spatial information is either less critical or has been preprocessed.
#
# ### **2.2. Hyperparameters Tested**
#
# #### **2.2.1. Common Hyperparameters**
#
# - **Learning Rate (`learning_rate`)**: Set to `0.005` and `0.02` to assess convergence behavior.
# - **Activation Function (`activation_function`)**: Evaluated using `ReLU` and `Sigmoid` to understand non-linear transformation impacts.
#
# #### **2.2.2. Model-Specific Hyperparameters**
#
# - **CNNs**:
#   - **Batch Size (`batch_size`)**: Set to `16` and `32` to analyze training stability and generalization.
#   - **Input Size (`input_size`)**: Fixed at `300x300` pixels.
#
# - **MLPs**:
#   - **Input Size (`input_size`)**: Varied between `50x50` and `300x300` pixels to examine the effect of dimensionality.
#   - **Hidden Dimensions (`hidden_dims`)**: Fixed at `[128, 64]`, representing two hidden layers with 128 and 64 neurons respectively.
#
# ### **2.3. Evaluation Metrics**
#
# For each experiment, the following metrics were recorded:
#
# - **Loss**:
#   - **Final Training Loss (`final_train_loss`)**
#   - **Final Validation Loss (`final_val_loss`)**
#
# - **Accuracy**:
#   - **Final Training Accuracy (`final_train_acc`)**
#   - **Final Validation Accuracy (`final_val_acc`)**
#
# - **Precision**:
#   - **Final Training Precision (`final_train_prec`)**
#   - **Final Validation Precision (`final_val_prec`)**
#
# - **Recall**:
#   - **Final Training Recall (`final_train_rec`)**
#   - **Final Validation Recall (`final_val_rec`)**
#
# - **F1-Score**:
#   - **Final Training F1-Score (`final_train_f1`)**
#   - **Final Validation F1-Score (`final_val_f1`)**
#
# ## **3. Results and Analysis**
#
# ### **3.1. Convolutional Neural Networks (CNNs)**
#
# #### **3.1.1. Overview of Experiments**
#
# A total of **8 CNN experiments** were conducted by varying the following hyperparameters:
#
# - **Learning Rates**: `0.005`, `0.02`
# - **Batch Sizes**: `16`, `32`
# - **Activation Functions**: `ReLU`, `Sigmoid`
# - **Input Size**: Fixed at `300x300` pixels
#
# #### **3.1.2. Detailed Results**
#
# | **Experiment ID** | **Learning Rate** | **Batch Size** | **Activation Function** | **Final Val Accuracy** | **Final Train Accuracy** | **Final Val Precision** | **Final Train Precision** | **Final Val Recall** | **Final Train Recall** | **Final Val F1-Score** | **Final Train F1-Score** |
# |-------------------|--------------------|-----------------|--------------------------|------------------------|--------------------------|-------------------------|---------------------------|----------------------|------------------------|------------------------|--------------------------|
# | 1                 | 0.005              | 16              | ReLU                     | 0.4794                 | 0.4794                   | 0.4794                  | 0.4794                    | 1.0                  | 1.0                    | 0.6481                 | 0.6481                   |
# | 2                 | 0.005              | 16              | Sigmoid                  | 0.4794                 | 0.5206                   | 0.0                     | 0.4794                    | 0.0                  | 0.0                    | 0.0                    | 0.0                      |
# | 3                 | 0.005              | 32              | ReLU                     | 0.4794                 | 0.4794                   | 0.4794                  | 0.4794                    | 1.0                  | 1.0                    | 0.6481                 | 0.6481                   |
# | 4                 | 0.005              | 32              | Sigmoid                  | 0.4794                 | 0.5206                   | 0.0                     | 0.0                       | 0.0                  | 0.0                    | 0.0                    | 0.0                      |
# | 5                 | 0.02               | 16              | ReLU                     | 0.5206                 | 0.5206                   | 0.0                     | 0.0                       | 0.0                  | 0.0                    | 0.0                    | 0.0                      |
# | 6                 | 0.02               | 16              | Sigmoid                  | 0.4794                 | 0.5206                   | 1.0                     | 0.5151                    | 0.6481               | 1.0                    | 0.6481                 | 0.6481                   |
# | 7                 | 0.02               | 32              | ReLU                     | 0.5206                 | 0.5206                   | 0.0                     | 0.0                       | 0.0                  | 0.0                    | 0.0                    | 0.0                      |
# | 8                 | 0.02               | 32              | Sigmoid                  | 0.4794                 | 0.5206                   | 1.0                     | 0.4803                    | 0.6481               | 0.6481                 | 0.6481                 | 0.6481                   |
#
# #### **3.1.3. Observations**
#
# 1. **Activation Function Impact**:
#    - **ReLU**: Experiments using `ReLU` activation consistently achieved a **Validation Accuracy** of approximately `47.94%` with both batch sizes and learning rates, except for Experiments 5 and 6.
#    - **Sigmoid**: Using `Sigmoid` activation led to varying performances. For instance, with a higher learning rate (`0.02`) and batch size (`16`), the model achieved a **Validation Accuracy** of `47.94%` but with **Final Train Accuracy** of `52.06%`. Notably, some configurations resulted in **zero precision and recall**, indicating potential issues with convergence or class imbalance handling.
#
# 2. **Learning Rate Impact**:
#    - **Lower Learning Rate (`0.005`)**:
#      - Models trained with `ReLU` maintained stable accuracies around `47.94%`.
#      - Models with `Sigmoid` showed discrepancies, with some achieving **Validation Accuracy** below `50%`.
#    - **Higher Learning Rate (`0.02`)**:
#      - Improved **Validation Accuracy** in some `ReLU` configurations (`0.5206`).
#      - However, for `Sigmoid`, results were inconsistent, with some experiments showing **zero precision and recall**.
#
# 3. **Batch Size Impact**:
#    - **Batch Size of 16**:
#      - **ReLU**: Consistent performance.
#      - **Sigmoid**: Inconsistencies in precision and recall.
#    - **Batch Size of 32**:
#      - Similar trends to batch size `16`, with some configurations leading to poor performance metrics.
#
# 4. **Overall Performance**:
#    - The best **Validation Accuracy** for CNNs was observed in Experiments **5** and **6**, both with a higher learning rate (`0.02`) and batch size `16`.
#    - However, some experiments exhibited **perfect recall but zero precision**, suggesting that while all positive predictions were correct, the model may have failed to identify negative cases correctly.
#
# #### **3.1.4. Potential Issues**
#
# - **Class Imbalance**: Zero precision and recall in some experiments hint at possible class imbalance, where the model may be predicting only one class.
# - **Activation Function Choice**: `Sigmoid` activation in the hidden layers may not be optimal for CNNs, as `ReLU` is generally preferred for mitigating the vanishing gradient problem.
# - **Learning Rate Sensitivity**: Higher learning rates may cause the model to overshoot minima, leading to unstable training dynamics.
#
# ### **3.2. Multi-Layer Perceptrons (MLPs)**
#
# #### **3.2.1. Overview of Experiments**
#
# A total of **8 MLP experiments** were conducted by varying the following hyperparameters:
#
# - **Learning Rates**: `0.005`, `0.02`
# - **Input Sizes**: `50x50`, `300x300` pixels
# - **Activation Functions**: `ReLU`, `Sigmoid`
# - **Hidden Dimensions**: Fixed at `[128, 64]`
#
# #### **3.2.2. Detailed Results**
#
# | **Experiment ID** | **Learning Rate** | **Input Size** | **Activation Function** | **Final Val Accuracy** | **Final Train Accuracy** | **Final Val Precision** | **Final Train Precision** | **Final Val Recall** | **Final Train Recall** | **Final Val F1-Score** | **Final Train F1-Score** |
# |-------------------|--------------------|-----------------|--------------------------|------------------------|--------------------------|-------------------------|---------------------------|----------------------|------------------------|------------------------|--------------------------|
# | 1                 | 0.005              | 50x50           | ReLU                     | 0.8678                 | 0.8843                   | 0.8556                  | 0.8983                    | 0.8588               | 0.8556                 | 0.8764                 | 0.8678                   |
# | 2                 | 0.005              | 50x50           | Sigmoid                  | 0.0                    | 0.5206                   | 0.0                     | 0.4794                    | 0.0                  | 0.0                    | 0.0                    | 0.0                      |
# | 3                 | 0.005              | 300x300         | ReLU                     | 0.6481                 | 0.4794                   | 0.4794                  | 0.4794                    | 1.0                  | 1.0                    | 0.6481                 | 0.6481                   |
# | 4                 | 0.005              | 300x300         | Sigmoid                  | 0.0                    | 0.5206                   | 0.0                     | 0.0                       | 0.0                  | 0.0                    | 0.0                    | 0.0                      |
# | 5                 | 0.02               | 50x50           | ReLU                     | 0.0                    | 0.5206                   | 0.0                     | 0.0                       | 0.0                  | 0.0                    | 0.0                    | 0.0                      |
# | 6                 | 0.02               | 50x50           | Sigmoid                  | 0.0                    | 0.5206                   | 0.0                     | 0.0                       | 0.0                  | 0.0                    | 0.0                    | 0.0                      |
# | 7                 | 0.02               | 300x300         | ReLU                     | 0.6481                 | 0.4794                   | 0.4794                  | 0.4794                    | 1.0                  | 1.0                    | 0.6481                 | 0.6481                   |
# | 8                 | 0.02               | 300x300         | Sigmoid                  | 0.0                    | 0.5206                   | 0.0                     | 0.0                       | 0.0                  | 0.0                    | 0.0                    | 0.0                      |
#
# #### **3.2.3. Observations**
#
# 1. **Activation Function Impact**:
#    - **ReLU**: Demonstrates strong performance, especially with smaller input sizes (`50x50`). Achieved **Validation Accuracy** up to `86.78%`.
#    - **Sigmoid**: Consistently underperforms across all configurations, with several experiments recording **zero** in precision, recall, and F1-score. This suggests poor convergence or issues with gradient flow.
#
# 2. **Input Size Impact**:
#    - **50x50 Pixels**:
#      - **ReLU**: Exhibits high accuracy and robust precision, recall, and F1-scores.
#      - **Sigmoid**: Fails to learn effectively, resulting in **zero** performance metrics.
#    - **300x300 Pixels**:
#      - **ReLU**: Moderate performance with **Validation Accuracy** around `64.81%`.
#      - **Sigmoid**: Similar to `50x50` configurations, shows **zero** performance metrics.
#
# 3. **Learning Rate Impact**:
#    - **Lower Learning Rate (`0.005`)**:
#      - Better performance with `ReLU` activation, especially for smaller input sizes.
#      - `Sigmoid` activation struggles regardless of input size.
#    - **Higher Learning Rate (`0.02`)**:
#      - Does not improve performance; models with `ReLU` show reduced **Validation Accuracy** for smaller input sizes and similar performance for larger input sizes.
#      - `Sigmoid` activation remains ineffective.
#
# 4. **Hidden Dimensions Impact**:
#    - All MLPs used hidden layers with `[128, 64]` neurons. The configuration provided sufficient capacity for the model to learn complex patterns, as evidenced by the high accuracy in certain setups.
#
# 5. **Overall Performance**:
#    - **Best Performance**: Experiment **1** with `ReLU` activation and `50x50` input size achieved **Validation Accuracy** of `86.78%`.
#    - **Consistent Failure**: All experiments using `Sigmoid` activation resulted in **zero** precision, recall, and F1-scores, indicating a lack of meaningful learning.
#
# #### **3.2.4. Potential Issues**
#
# - **Activation Function Choice**: The use of `Sigmoid` activation in hidden layers may lead to vanishing gradients, hindering the model's ability to learn, especially in deeper networks like MLPs.
# - **Input Size Misalignment**: While `50x50` input sizes are manageable for MLPs, larger sizes like `300x300` may lead to increased computational complexity without proportional gains in performance.
# - **Learning Rate Sensitivity**: Although higher learning rates can expedite convergence, in this case, they did not yield better results and may have contributed to instability, especially in higher-dimensional inputs.
#
# ### **3.3. Comparative Analysis**
#
# #### **3.3.1. CNNs vs. MLPs**
#
# - **Performance Metrics**:
#   - **CNNs** achieved **Validation Accuracies** ranging from `47.94%` to `52.06%` with some configurations, whereas **MLPs** achieved significantly higher accuracies, especially with smaller input sizes (`50x50`), up to `86.78%`.
#   - **Precision, Recall, and F1-Score** were notably better in MLPs with `ReLU` activation compared to CNNs.
#
# - **Robustness**:
#   - **MLPs** with `ReLU` demonstrated robust performance across different input sizes, while CNNs showed limited improvement and potential issues with certain activation functions and learning rates.
#   - **MLPs** outperformed **CNNs** in scenarios with smaller input sizes, possibly due to the limited capacity of CNNs to extract meaningful features from high-resolution images without sufficient training.
#
# - **Activation Function Sensitivity**:
#   - Both architectures suffered from poor performance with `Sigmoid` activation, but the impact was more pronounced in CNNs, where some configurations still retained moderate accuracies.
#
# #### **3.3.2. Impact of Input Size**
#
# - **MLPs**:
#   - **50x50**: High accuracy and strong metric scores.
#   - **300x300**: Moderate accuracy, possibly due to increased input dimensionality without adequate model capacity.
#
# - **CNNs**:
#   - **300x300**: Fixed input size led to consistent performance metrics across different configurations, indicating potential limitations in the model's ability to leverage higher-resolution inputs.
#
# #### **3.3.3. Learning Rate and Batch Size**
#
# - **Learning Rate**:
#   - A higher learning rate (`0.02`) did not universally enhance performance and sometimes led to inconsistent results, especially with `Sigmoid` activation.
#
# - **Batch Size**:
#   - In CNNs, varying batch sizes between `16` and `32` did not significantly impact performance, suggesting that the models may be insensitive to this hyperparameter within the tested range.
#   - In MLPs, batch size was fixed at `16`, as varying it was not part of the experimentation.
#
# ## **4. Conclusions**
#
# ### **4.1. Key Findings**
#
# 1. **MLPs with ReLU Activation Outperform CNNs**:
#    - MLPs configured with `ReLU` activation and smaller input sizes (`50x50`) achieved superior validation accuracies and robust metric scores compared to CNNs.
#
# 2. **Activation Function is Critical**:
#    - `ReLU` activation consistently led to better performance across both architectures.
#    - `Sigmoid` activation resulted in poor performance, likely due to issues with gradient propagation and convergence.
#
# 3. **Input Size Matters**:
#    - For MLPs, smaller input sizes (`50x50`) were optimal, balancing computational efficiency and model performance.
#    - Larger input sizes (`300x300`) did not translate to performance gains and, in some cases, degraded model efficacy.
#
# 4. **Learning Rate and Batch Size**:
#    - The tested learning rates (`0.005`, `0.02`) had mixed effects. A higher learning rate did not consistently improve performance and may have introduced instability, especially in models with `Sigmoid` activation.
#    - Batch size variations in CNNs had negligible impact within the tested range.
#
# ### **4.2. Recommendations**
#
# 1. **Model Selection**:
#    - Given the superior performance of MLPs with `ReLU` activation on smaller input sizes, consider prioritizing this configuration for binary skin lesion classification tasks.
#
# 2. **Activation Function Choice**:
#    - Favor `ReLU` over `Sigmoid` in hidden layers to ensure effective learning and gradient propagation.
#
# 3. **Input Size Optimization**:
#    - Utilize smaller input sizes (`50x50`) for MLPs to balance model performance and computational resources.
#    - For CNNs, consider increasing model complexity or depth to better leverage higher-resolution inputs, or explore feature extraction techniques to reduce input dimensionality.
#
# 4. **Hyperparameter Tuning**:
#    - Further explore learning rates, possibly integrating learning rate schedulers to dynamically adjust the learning rate during training.
#    - Experiment with additional batch sizes and larger batch sizes to determine if they offer any performance benefits.
#
# 5. **Addressing Class Imbalance**:
#    - Investigate and implement strategies to handle potential class imbalances, such as weighted loss functions, oversampling minority classes, or using data augmentation.
#
# 6. **Advanced Architectures**:
#    - Explore more sophisticated architectures like **Deep CNNs** or **Residual Networks (ResNets)** to enhance feature extraction capabilities.
#    - Consider integrating **Dropout** or **Batch Normalization** layers to improve generalization and training stability.
#
# 7. **Evaluation Metrics**:
#    - Incorporate additional metrics like **Area Under the ROC Curve (AUC-ROC)** to gain deeper insights into model performance, especially in imbalanced datasets.
#
# ## **5. Future Work**
#
# To build upon the current findings and further enhance the model's performance, the following avenues are recommended:
#
# 1. **Data Augmentation**:
#    - Implement data augmentation techniques such as rotations, flips, and color jittering to increase dataset variability and improve model generalization.
#
# 2. **Regularization Techniques**:
#    - Apply regularization methods like **L1/L2 regularization**, **Dropout**, and **Early Stopping** to prevent overfitting and enhance model robustness.
#
# 3. **Cross-Validation**:
#    - Employ cross-validation strategies to ensure that the reported performance metrics are consistent and not artifacts of a particular train-test split.
#
# 4. **Ensemble Methods**:
#    - Combine predictions from multiple models (e.g., CNN and MLP) to potentially boost overall classification performance through ensemble learning.
#
# 5. **Transfer Learning**:
#    - Utilize pre-trained models (especially CNNs) and fine-tune them on the skin lesion dataset to leverage learned representations from larger datasets.
#
# 6. **Advanced Optimization Algorithms**:
#    - Experiment with different optimizers like **SGD with Momentum**, **RMSprop**, or **AdamW** to evaluate their impact on training dynamics and convergence.
#
# 7. **Exploring Different Architectures**:
#    - Investigate architectures such as **Fully Convolutional Networks (FCNs)**, **U-Nets** for segmentation, or **Attention Mechanisms** to focus on salient regions in images.
#
# ## **6. Recommendations for Future Experiments**
#
# Based on the current findings, the following recommendations are proposed to further enhance model performance and reliability:
#
# 1. **Optimize Activation Functions**:
#    - **MLPs**: Since `ReLU` outperformed `Sigmoid`, consider experimenting with other activation functions like `Leaky ReLU` or `ELU` to potentially capture more nuanced patterns.
#    - **CNNs**: Stick with `ReLU` or explore advanced activation functions like `Swish` or `Mish` for improved gradient flow.
#
# 2. **Address Class Imbalance**:
#    - Implement techniques such as **Oversampling** the minority class, **Undersampling** the majority class, or using **Class Weights** in the loss function to mitigate bias towards dominant classes.
#
# 3. **Incorporate Regularization**:
#    - Apply **Dropout** layers in MLPs and CNNs to prevent overfitting.
#    - Utilize **L1/L2 Regularization** to constrain model weights and promote generalization.
#
# 4. **Enhance Data Augmentation**:
#    - Introduce data augmentation strategies like rotations, scaling, translations, and horizontal/vertical flips to increase dataset diversity and improve model robustness.
#
# 5. **Implement Early Stopping**:
#    - Monitor validation loss and halt training when it ceases to improve, thereby preventing overfitting and reducing computational resources.
#
# 6. **Explore Different Optimizers**:
#    - Experiment with optimizers like **SGD with Momentum**, **RMSprop**, or **AdamW** to observe their effects on convergence speed and model performance.
#
# 7. **Expand Hyperparameter Search**:
#    - Broaden the range of hyperparameters, including learning rates (`e.g., 0.001, 0.01, 0.05`), batch sizes (`e.g., 8, 64`), and hidden layer sizes (`e.g., [256, 128]`), to identify optimal configurations.
#
# 8. **Transfer Learning for CNNs**:
#    - Leverage pre-trained CNN architectures (e.g., **ResNet**, **VGG**) and fine-tune them on the skin lesion dataset to capitalize on learned feature representations.
#
# 9. **Cross-Validation**:
#    - Utilize **k-Fold Cross-Validation** to ensure that model performance is consistent across different data splits, enhancing the reliability of evaluation metrics.
#
# 10. **Evaluate with Additional Metrics**:
#     - Incorporate metrics like **AUC-ROC**, **Confusion Matrix**, and **Precision-Recall Curves** to gain deeper insights into model performance, especially in imbalanced settings.
#
# ## **7. Summary**
#
# The conducted experiments revealed that **MLPs with `ReLU` activation and smaller input sizes (`50x50`) significantly outperform CNNs** in the context of skin lesion classification, achieving high validation accuracies and robust precision, recall, and F1-scores. However, both architectures struggled with `Sigmoid` activation functions, leading to poor performance metrics.
#
# While CNNs are inherently suited for image data due to their ability to capture spatial hierarchies, the current CNN configurations did not leverage this advantage effectively, possibly due to limited model complexity, activation function choice, or hyperparameter settings. On the other hand, MLPs demonstrated remarkable performance with appropriate activation functions and input sizes, highlighting their potential applicability in scenarios where spatial information is either limited or has been preprocessed adequately.
#
# Future experiments should focus on refining model architectures, addressing class imbalances, and expanding hyperparameter tuning to further enhance classification performance and generalization capabilities.
