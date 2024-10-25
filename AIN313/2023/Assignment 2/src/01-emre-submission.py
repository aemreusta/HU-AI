# %%
import os
import numpy as np
import concurrent.futures
import logging
import pickle
import time
import matplotlib.pyplot as plt
import pandas as pd

# %%
# Directories
working_dir = "/Users/emre/GitHub/HU-AI/AIN313/Assignment 2"
DATASET_PATH = os.path.join(working_dir, "dataset")
PROCESSED_DATASET_PATH = os.path.join(DATASET_PATH, "processed")
MODELS_PATH = os.path.join(working_dir, "models")
GRAPH_PATH = os.path.join(working_dir, "graphs")
OUTPUTS_PATH = os.path.join(working_dir, "outputs")

# Raw Dataset
RAW_DATASET_PATH = os.path.join(DATASET_PATH, "raw")
CLASSES_PATH = os.path.join(RAW_DATASET_PATH, "RIT_18", "classes.npy")
TRAIN_DATA_PATH = os.path.join(RAW_DATASET_PATH, "RIT_18", "train_data.npy")
TRAIN_MASK_PATH = os.path.join(RAW_DATASET_PATH, "RIT_18", "train_mask.npy")
TRAIN_LABEL_PATH = os.path.join(RAW_DATASET_PATH, "RIT_18", "train_labels.npy")

# %% [markdown]
# # EDA

# %%
classes = np.load(CLASSES_PATH)
train_data = np.load(TRAIN_DATA_PATH)
train_mask = np.load(TRAIN_MASK_PATH)
train_labels = np.load(TRAIN_LABEL_PATH)

# %%
classes

# %%
print(f"Train data shape: {train_data.shape}")
print(f"Train mask shape: {train_mask.shape}")
print(f"Train labels shape: {train_labels.shape}")

# %%
# Assuming 'train_data' contains the multispectral image with six VNIR spectral bands
variances = np.var(train_data, axis=(0, 1))

# sort the variances in descending order
# variances = np.sort(variances)[::-1]

# Print the indices and variances of the bands
for i, variance in enumerate(variances):
    print(f"Band {i + 1}: Variance = {variance}")

# %%
# Assuming 'train_data' contains the multispectral image with six VNIR spectral bands

# Extract the most informative three spectral bands for RGB
rgb_image = train_data[:, :, 3:]

# Normalize pixel values to the range [0, 1] for each channel independently
rgb_image_normalized = (rgb_image - np.min(rgb_image, axis=(0, 1))) / (
    np.max(rgb_image, axis=(0, 1)) - np.min(rgb_image, axis=(0, 1))
)

# Apply gamma correction for brightness enhancement
gamma = 3.2
rgb_image_normalized_gamma = np.power(rgb_image_normalized, 1 / gamma)

# Display the original and enhanced RGB images side by side
plt.figure(figsize=(12, 10))

# Original RGB Image
plt.subplot(3, 3, 1)
plt.imshow(rgb_image)
plt.title("Original RGB Image")
plt.axis("off")  # Hide the axis ticks

# Enhanced RGB Image
plt.subplot(3, 3, 2)
plt.imshow(rgb_image_normalized)
plt.title("Enhanced RGB Image")
plt.axis("off")  # Hide the axis ticks

# Display the enhanced RGB image with gamma correction
plt.subplot(3, 3, 3)
plt.imshow(rgb_image_normalized_gamma)
plt.title("Enhanced RGB Image with Gamma Correction")
plt.axis("off")  # Hide the axis ticks

# show first infared chanel
plt.subplot(3, 3, 4)
plt.imshow(train_data[:, :, 0])
plt.title("First Infared Channel")
plt.axis("off")  # Hide the axis ticks

# show second infared chanel
plt.subplot(3, 3, 5)
plt.imshow(train_data[:, :, 1])
plt.title("Second Infared Channel")
plt.axis("off")  # Hide the axis ticks

# show third infared chanel
plt.subplot(3, 3, 6)
plt.imshow(train_data[:, :, 2])
plt.title("Third Infared Channel")
plt.axis("off")  # Hide the axis ticks

# show masks
plt.subplot(3, 3, 7)
plt.imshow(train_mask)
plt.title("Train Mask")
plt.axis("off")  # Hide the axis ticks

# show labels
plt.subplot(3, 3, 8)
plt.imshow(train_labels)
plt.title("Train Labels")
plt.axis("off")  # Hide the axis ticks

plt.tight_layout()
# save the figure
plt.savefig(os.path.join(GRAPH_PATH, "01-data-with-channels.png"))
plt.show()

# %%
# show train_labels and add colorbar
# Display the original and enhanced RGB images side by side
plt.figure(figsize=(8, 5))
plt.imshow(train_labels)

# Use class indices as ticks
ticks = np.arange(len(classes))

# Set class names as tick labels
plt.colorbar(ticks=ticks, label="Class")

plt.title("Train Labels")
plt.axis("off")  # Hide the axis ticks
plt.tight_layout()

# Save the figure
plt.savefig(os.path.join(GRAPH_PATH, "02-train-labels.png"))
plt.show()

# %%
unique, count = np.unique(train_mask, return_counts=True)

for i, (u, c) in enumerate(zip(unique, count)):
    print(f"Class {u}: {c} pixels")

# %%
classes_dict = {
    "0": "Other Class/Image Border",
    "1": "Road Markings, Asphalt, Landing Pad",
    "2": "Water",
    "3": "Building",
    "4": "Vehicle (Car, Truck, or Bus)",
    "5": "Person",
    "6": "Vegetation",
    "7": "Wood Panel",
    "8": "Rocks, Sand",
    "9": "Chair, Table",
}

# %%
unique, count = np.unique(train_labels, return_counts=True)

for i, (u, c) in enumerate(zip(unique, count)):
    # print(classes_dict[str(u)])
    print(f"Class {u} -> {classes_dict[str(u)]}: {c} pixels")

# %%
# Extract the most informative three spectral bands for RGB
rgb_image = train_data[:, :, 3:]

# Normalize pixel values to the range [0, 1] for each channel independently
rgb_image_normalized = (rgb_image - np.min(rgb_image, axis=(0, 1))) / (
    np.max(rgb_image, axis=(0, 1)) - np.min(rgb_image, axis=(0, 1))
)

# Set the percentage for the test split
test_split_percentage = 30

# Calculate the index to split the data
split_index = int((test_split_percentage / 100) * rgb_image_normalized.shape[1])
# print(f'Split index: {split_index}')

# Split the data into training and testing sets
train_rgb = rgb_image_normalized[:, :-split_index, :]
test_rgb = rgb_image_normalized[:, -split_index:, :]

# Display the shape of the training and testing sets
print(f"Train RGB shape: {train_rgb.shape}")
print(f"Test RGB shape: {test_rgb.shape}")

# Assuming rgb_image_normalized is your 3D array
flat_train_rgb = np.reshape(train_rgb, (-1, 3))
flat_test_rgb = np.reshape(test_rgb, (-1, 3))

# save them to disk as npy file
np.save(
    os.path.join(PROCESSED_DATASET_PATH, "train_rgb.npy"),
    flat_train_rgb.astype(np.float32),
)
np.save(
    os.path.join(PROCESSED_DATASET_PATH, "test_rgb.npy"),
    flat_test_rgb.astype(np.float32),
)

# Display the shape of the flattened training and testing sets
print(f"Flat Train RGB shape: {flat_train_rgb.shape}")
print(f"Flat Test RGB shape: {flat_test_rgb.shape}")

# %%
# show the train and test rgb images
plt.figure(figsize=(12, 10))

plt.subplot(1, 2, 1)
plt.imshow(train_rgb)
plt.title("Train RGB Image")
plt.axis("off")  # Hide the axis ticks

plt.subplot(1, 2, 2)
plt.imshow(test_rgb)
plt.title("Test RGB Image")
plt.axis("off")  # Hide the axis ticks


plt.tight_layout()
# save the figure
plt.savefig(os.path.join(GRAPH_PATH, "03-train-test-rgb.png"))
plt.show()

# %%
infra_image = train_data[:, :, :3]

# Normalize pixel values to the range [0, 1] for each channel independently
infra_image_normalized = (infra_image - np.min(infra_image, axis=(0, 1))) / (
    np.max(infra_image, axis=(0, 1)) - np.min(infra_image, axis=(0, 1))
)

# Split the data into training and testing sets
train_infra = infra_image_normalized[:, :-split_index, :]
test_infra = infra_image_normalized[:, -split_index:, :]

# Assuming rgb_image_normalized is your 3D array
flat_train_infra = np.reshape(train_infra, (-1, 3))
flat_test_infra = np.reshape(test_infra, (-1, 3))

# save them to disk as npy file
np.save(
    os.path.join(PROCESSED_DATASET_PATH, "train_infra.npy"),
    flat_train_infra.astype(np.float32),
)
np.save(
    os.path.join(PROCESSED_DATASET_PATH, "test_infra.npy"),
    flat_test_infra.astype(np.float32),
)

# Display the shape of the training and testing sets
print(f"Train Infra shape: {flat_train_infra.shape}")
print(f"Test Infra shape: {flat_test_infra.shape}")

# %%
plt.figure(figsize=(12, 10))

plt.subplot(1, 2, 1)
plt.imshow(train_infra)
plt.title("Train Infra Image")
plt.axis("off")  # Hide the axis ticks

plt.subplot(1, 2, 2)
plt.imshow(test_infra)
plt.title("Test Infra Image")
plt.axis("off")  # Hide the axis ticks

plt.tight_layout()
# save the figure
plt.savefig(os.path.join(GRAPH_PATH, "04-train-test-infra.png"))
plt.show()

# %%
whole_image = train_data

# Normalize pixel values to the range [0, 1] for each channel independently
whole_image_normalized = (whole_image - np.min(whole_image, axis=(0, 1))) / (
    np.max(whole_image, axis=(0, 1)) - np.min(whole_image, axis=(0, 1))
)

# Split the data into training and testing sets
train_whole = whole_image_normalized[:, :-split_index, :]
test_whole = whole_image_normalized[:, -split_index:, :]

# Assuming rgb_image_normalized is your 3D array
flat_train_whole = np.reshape(train_whole, (-1, 6))
flat_test_whole = np.reshape(test_whole, (-1, 6))

# save them to disk as npy file
np.save(
    os.path.join(PROCESSED_DATASET_PATH, "train_whole.npy"),
    flat_train_whole.astype(np.float32),
)
np.save(
    os.path.join(PROCESSED_DATASET_PATH, "test_whole.npy"),
    flat_test_whole.astype(np.float32),
)

# Display the shape of the training and testing sets
print(f"Train Whole shape: {flat_train_whole.shape}")
print(f"Test Whole shape: {flat_test_whole.shape}")

# %%
train_labels_divided = train_labels[:, :-split_index]
test_labels = train_labels[:, -split_index:]

flat_train_labels_divided = np.reshape(train_labels_divided, (-1, 1))
flat_test_labels = np.reshape(test_labels, (-1, 1))

# Save them to disk as npy file with dtype=int8
np.save(
    os.path.join(PROCESSED_DATASET_PATH, "train_labels.npy"),
    flat_train_labels_divided.astype(np.int8),
)
np.save(
    os.path.join(PROCESSED_DATASET_PATH, "test_labels.npy"),
    flat_test_labels.astype(np.int8),
)

# Display the shape of the training and testing sets
print(f"Train Labels shape: {flat_train_labels_divided.shape}")
print(f"Test Labels shape: {flat_test_labels.shape}")

# %%
plt.figure(figsize=(12, 10))

plt.subplot(1, 2, 1)
plt.imshow(train_labels_divided)
plt.title("Train Labels")
plt.axis("off")  # Hide the axis ticks

plt.subplot(1, 2, 2)
plt.imshow(test_labels)
plt.title("Test Labels")
plt.axis("off")  # Hide the axis ticks

plt.tight_layout()
# save the figure
plt.savefig(os.path.join(GRAPH_PATH, "05-train-test-labels.png"))

# %% [markdown]
# # Model Training

# %%
train_labels = np.load(os.path.join(PROCESSED_DATASET_PATH, "train_labels.npy"))
test_labels = np.load(os.path.join(PROCESSED_DATASET_PATH, "test_labels.npy"))

experiment_dict = {
    "train_rgb": os.path.join(PROCESSED_DATASET_PATH, "train_rgb.npy"),
    "test_rgb": os.path.join(PROCESSED_DATASET_PATH, "test_rgb.npy"),
    "train_infra": os.path.join(PROCESSED_DATASET_PATH, "train_infra.npy"),
    "test_infra": os.path.join(PROCESSED_DATASET_PATH, "test_infra.npy"),
    "train_whole": os.path.join(PROCESSED_DATASET_PATH, "train_whole.npy"),
    "test_whole": os.path.join(PROCESSED_DATASET_PATH, "test_whole.npy"),
}


# %%
class NaiveBayesClassifier:
    def __init__(
        self,
        verbose=False,
        multithreading=False,
        num_threads=None,
        save_model=False,
        model_file=None,
        batch_size=1000,
    ):
        self.verbose = verbose
        self.class_summary = None
        self.multithreading = multithreading
        self.num_threads = num_threads
        self.save_model = save_model
        self.model_file = model_file
        self.batch_size = batch_size
        self.logger = self._setup_logger()

        """
        Initialize NaiveBayesClassifier.

        Parameters:
        - verbose: Enable verbose logging.
        - multithreading: Use multithreading for parallel processing.
        - num_threads: Number of threads for parallel processing.
        - save_model: Save the trained model to a file.
        - model_file: File path to save/load the model.
        - batch_size: Batch size for prediction.

        Attributes:
        - verbose: Verbose mode flag.
        - class_summary: Summary of each class during training.
        - multithreading: Multithreading flag.
        - num_threads: Number of threads for parallel processing.
        - save_model: Save model flag.
        - model_file: File path for saving/loading the model.
        - batch_size: Batch size for prediction.
        - logger: Logger for logging messages.
        """

    def _setup_logger(self):
        """
        Set up the logger for logging messages.
        """
        logger = logging.getLogger(__name__)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        logger.setLevel(logging.INFO)
        return logger

    def log(self, message):
        """
        Log a message using the logger.

        Parameters:
        - message: Message to be logged.
        """

        self.logger.info(message)

    def separate_classes(self, X, y):
        """
        Separate input data into classes.

        Parameters:
        - X: Input features.
        - y: Class labels.

        Returns:
        Dictionary with class labels as keys and corresponding feature values.
        """

        separated_classes = {}
        for i, class_name in enumerate(y):
            if class_name not in separated_classes:
                separated_classes[class_name] = []
            separated_classes[class_name].append(X[i])
        return separated_classes

    def stat_info(self, X):
        """
        Calculate statistics (mean and standard deviation) for each feature.

        Parameters:
        - X: Input features.

        Returns:
        List of dictionaries containing mean and standard deviation for each feature.
        """

        return [
            {"std": np.std(feature), "mean": np.mean(feature)} for feature in zip(*X)
        ]

    def fit(self, X, y):
        """
        Train the Naive Bayes classifier.

        Parameters:
        - X: Training data features.
        - y: Training data labels.

        Returns:
        Class summary after training.
        """

        start_time = time.time()

        self.log(f"Training the model using {self.num_threads} thread(s)...")
        separated_classes = self.separate_classes(X, y)
        self.class_summary = {}

        executor_class = (
            concurrent.futures.ThreadPoolExecutor
            if self.multithreading
            else concurrent.futures.ProcessPoolExecutor
        )
        with executor_class(max_workers=self.num_threads) as executor:
            futures = {
                executor.submit(
                    self._fit_class, class_name, feature_values, X
                ): class_name
                for class_name, feature_values in separated_classes.items()
            }

            for future in concurrent.futures.as_completed(futures):
                class_name = futures[future]
                self.class_summary[class_name] = future.result()

        elapsed_time = time.time() - start_time
        self.log(f"Training completed in {elapsed_time:.2f} seconds.")

        if self.save_model:
            self.save_model_to_file()

        return self.class_summary

    def _fit_class(self, class_name, feature_values, X):
        """
        Fit a specific class during training.

        Parameters:
        - class_name: Name of the class.
        - feature_values: Feature values for the class.
        - X: Training data features.

        Returns:
        Dictionary containing prior probability and summary statistics for the class.
        """

        result = {
            "prior_proba": len(feature_values) / len(X),
            "summary": {
                "std": np.std(feature_values, axis=0),
                "mean": np.mean(feature_values, axis=0),
            },
        }
        self.log(f"Class {class_name} trained.")
        return result

    def distribution(self, x, mean, std):
        """
        Compute the Gaussian distribution.

        Parameters:
        - x: Input value.
        - mean: Mean of the distribution.
        - std: Standard deviation of the distribution.

        Returns:
        Gaussian distribution value.
        """

        exponent = np.exp(-((x - mean) ** 2) / (2 * std**2))
        return exponent / (np.sqrt(2 * np.pi) * std)

    def predict(self, X):
        """
        Make predictions for input data.

        Parameters:
        - X: Input data features.

        Returns:
        Array of predicted class labels.
        """

        start_time = time.time()

        if self.class_summary is None and self.model_file:
            self.load_model_from_file()

        self.log(
            f"Predicting the class using {self.num_threads} thread(s) and batch size {self.batch_size}..."
        )

        predictions = []

        executor_class = (
            concurrent.futures.ThreadPoolExecutor
            if self.multithreading
            else concurrent.futures.ProcessPoolExecutor
        )
        with executor_class(max_workers=self.num_threads) as executor:
            batch_size = self.batch_size
            num_batches = len(X) // batch_size
            remainder = len(X) % batch_size

            futures = {
                executor.submit(self._predict_batch, batch, i): i
                for i, batch in enumerate(np.array_split(X, num_batches))
            }

            if remainder > 0:
                last_batch_index = num_batches
                last_batch = X[last_batch_index * batch_size :]
                futures.update(
                    {
                        executor.submit(
                            self._predict_batch, last_batch, last_batch_index
                        ): last_batch_index
                    }
                )

            predictions = []
            actual_sizes = []
            for future in concurrent.futures.as_completed(futures):
                futures[future]
                pred_batch, actual_size = future.result()
                predictions.extend(pred_batch)
                actual_sizes.append(actual_size)

        elapsed_time = time.time() - start_time
        self.log(f"Prediction completed in {elapsed_time:.2f} seconds.")

        return np.array(predictions)[: sum(actual_sizes)]  # Adjust the slicing

    def _predict_batch(self, batch, batch_index):
        """
        Make predictions for a batch of input data.

        Parameters:
        - batch: Batch of input data features.
        - batch_index: Index of the batch.

        Returns:
        Tuple containing predictions and batch size.
        """

        self.log(f"Running prediction for batch {batch_index}...")
        predictions = np.concatenate([self._predict_row(row) for row in batch])
        return predictions, len(batch)  # Return the batch size

    def _predict_row(self, row):
        """
        Make predictions for a single row of input data.

        Parameters:
        - row: Input data row.

        Returns:
        Array of predicted class labels.
        """

        max_log_proba = float("-inf")
        predicted_class = None
        predictions = []

        for class_name, features in self.class_summary.items():
            log_likelihood = np.sum(
                np.log(
                    self.distribution(
                        row,
                        features["summary"]["mean"],
                        features["summary"]["std"] + 1e-8,  # Add a small constant
                    )
                )
            )
            log_posterior = np.log(features["prior_proba"]) + log_likelihood

            if log_posterior > max_log_proba:
                max_log_proba = log_posterior
                predicted_class = class_name

        predictions.append(predicted_class)

        return np.array(predictions)

    def accuracy(self, y_test, y_pred):
        """
        Calculate the accuracy of predictions.

        Parameters:
        - y_test: True class labels.
        - y_pred: Predicted class labels.

        Returns:
        Accuracy value.
        """

        if len(y_test) != len(y_pred):
            raise ValueError("Length of y_test and y_pred must be the same.")

        accuracy_value = np.sum(np.array(y_test) == np.array(y_pred)) / len(y_test)
        self.log(f"Accuracy: {accuracy_value}")
        return accuracy_value

    def save_model_to_file(self, file_path=None):
        """
        Save the trained model to a file.

        Parameters:
        - file_path: File path for saving the model.
        """

        if file_path is None:
            file_path = self.model_file

        if file_path:
            with open(file_path, "wb") as file:
                pickle.dump(self.class_summary, file)
                self.log(f"Model saved to {file_path}.")

    def load_model_from_file(self, file_path=None):
        """
        Load a trained model from a file.

        Parameters:
        - file_path: File path for loading the model.
        """

        if file_path is None:
            file_path = self.model_file

        if file_path:
            try:
                with open(file_path, "rb") as file:
                    self.class_summary = pickle.load(file)
                self.log(f"Model loaded from {file_path}.")
            except FileNotFoundError:
                self.log(f"No model file found at {file_path}. Training a new model.")
            except Exception as e:
                self.log(f"Error loading the model: {e}. Training a new model.")


# %% [markdown]
# ### RGB Model

# %%
# Load RGB training data
rgb_train_data = np.load(experiment_dict["train_rgb"])
print(
    "Train data shape:", rgb_train_data.shape, "Train data type:", rgb_train_data.dtype
)

# Display information about training labels
print(
    "Train labels shape:", train_labels.shape, "Train labels type:", train_labels.dtype
)

# Define paths and create an instance of the NaiveBayesClassifier for RGB data
rgb_model_path = os.path.join(MODELS_PATH, "naive_bayes_rgb.pkl")
rgb_model = NaiveBayesClassifier(
    verbose=True,
    multithreading=True,
    num_threads=8,
    save_model=True,
    model_file=rgb_model_path,
    batch_size=10000,
)

# Train the model if the model file doesn't exist
if not os.path.exists(rgb_model_path):
    rgb_model.fit(rgb_train_data, train_labels)

# Load RGB test data
rgb_test_data = np.load(experiment_dict["test_rgb"])
print("Test data shape:", rgb_test_data.shape, "Test data type:", rgb_test_data.dtype)

# Predict using the trained RGB model
rgb_test_pred = rgb_model.predict(rgb_test_data)
print("Test pred shape:", rgb_test_pred.shape, "Test pred type:", rgb_test_pred.dtype)

# Calculate and display RGB accuracy
rgb_acc = rgb_model.accuracy(test_labels, rgb_test_pred[: len(test_labels)])
print("RGB Accuracy:", rgb_acc)


# %% [markdown]
# ### Infra Model

# %%
# Load infrared training data
infra_train_data = np.load(experiment_dict["train_infra"])
print(
    "Train data shape:",
    infra_train_data.shape,
    "Train data type:",
    infra_train_data.dtype,
)

# Display information about training labels
print(
    "Train labels shape:", train_labels.shape, "Train labels type:", train_labels.dtype
)

# Define paths and create an instance of the NaiveBayesClassifier for infrared data
infra_model_path = os.path.join(MODELS_PATH, "naive_bayes_infra.pkl")
infra_model = NaiveBayesClassifier(
    verbose=True,
    multithreading=True,
    num_threads=8,
    save_model=True,
    model_file=infra_model_path,
    batch_size=10000,
)

# Train the model if the model file doesn't exist
if not os.path.exists(infra_model_path):
    infra_model.fit(infra_train_data, train_labels)

# Load infrared test data
infra_test_data = np.load(experiment_dict["test_infra"])
print(
    "Test data shape:", infra_test_data.shape, "Test data type:", infra_test_data.dtype
)

# Predict using the trained infrared model
infra_test_pred = infra_model.predict(infra_test_data)
print(
    "Test pred shape:", infra_test_pred.shape, "Test pred type:", infra_test_pred.dtype
)

# Calculate and display infrared accuracy
infra_acc = infra_model.accuracy(test_labels, infra_test_pred[: len(test_labels)])
print("Infra Accuracy:", infra_acc)


# %% [markdown]
# ### Whole Model

# %%
# Load whole training data
whole_train_data = np.load(experiment_dict["train_whole"])
print(
    "Train data shape:",
    whole_train_data.shape,
    "Train data type:",
    whole_train_data.dtype,
)

# Display information about training labels
print(
    "Train labels shape:", train_labels.shape, "Train labels type:", train_labels.dtype
)

# Define paths and create an instance of the NaiveBayesClassifier for whole data
whole_model_path = os.path.join(MODELS_PATH, "naive_bayes_whole.pkl")
whole_model = NaiveBayesClassifier(
    verbose=True,
    multithreading=True,
    num_threads=8,
    save_model=True,
    model_file=whole_model_path,
    batch_size=10000,
)

# Train the model if the model file doesn't exist
if not os.path.exists(whole_model_path):
    whole_model.fit(whole_train_data, train_labels)

# Load whole test data
whole_test_data = np.load(experiment_dict["test_whole"])
print(
    "Test data shape:", whole_test_data.shape, "Test data type:", whole_test_data.dtype
)

# Predict using the trained whole data model
whole_test_pred = whole_model.predict(whole_test_data)
print(
    "Test pred shape:", whole_test_pred.shape, "Test pred type:", whole_test_pred.dtype
)

# Calculate and display whole data accuracy
whole_acc = whole_model.accuracy(test_labels, whole_test_pred[: len(test_labels)])
print("Whole Accuracy:", whole_acc)

# %% [markdown]
# ### Show Results

# %%
# Create a DataFrame to store the results
df_results = pd.DataFrame(
    {
        "Model": ["RGB", "Infra", "Whole"],
        "Accuracy": [rgb_acc, infra_acc, whole_acc],
    }
)

# Save the DataFrame to a CSV file
df_results.to_csv(os.path.join(OUTPUTS_PATH, "naive_bayes_results.csv"), index=False)

# Plot the results as a bar plot with numbers over the bars
plt.figure(figsize=(10, 5))
plt.title("Naive Bayes Accuracy")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.grid()
bars = plt.bar(
    df_results["Model"], df_results["Accuracy"], color=["blue", "green", "red"]
)

# Add numbers over the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        yval,
        round(yval, 2),
        ha="center",
        va="bottom",
    )

plt.tight_layout()

# Save the plot
plt.savefig(os.path.join(GRAPH_PATH, "07-naive_bayes_accuracy_barplot.png"))
plt.show()
