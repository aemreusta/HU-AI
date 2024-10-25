# Project Title: Comparative Analysis of Custom CNN and EfficientNet for Image Classification on the Animals-10 Dataset

## Project Description
This project explores the application of convolutional neural networks (CNNs) for image classification, comparing the performance of a custom-built CNN and a pre-trained EfficientNet model on the Animals-10 dataset. The goal is to evaluate the effectiveness, computational efficiency, and generalization capability of these models in classifying images into 10 distinct animal categories.

## Code Organization

### Directories and Files
- `src/`: Contains all the source code files.
  - `train.py`: Script to train the models.
  - `evaluate.py`: Script for evaluating the models on the test dataset.
  - `model.py`: Contains the definitions of the CustomCNN and EfficientNet models.
  - `utils.py`: Helper functions for data loading, preprocessing, and augmentation.
- `data/`: Directory where the Animals-10 dataset is stored.
- `models/`: Saved models after training.
- `results/`: Graphs and result files from model evaluations.
- `logs/`: Training logs.

## Installation and Setup

### Prerequisites
- Python 3.8 or later
- PyTorch 1.7 or later
- torchvision
- matplotlib
- sklearn

### Setup
1. Clone the repository to your local machine.
2. Ensure that Python and pip are installed.
3. Install the required dependencies:
   ```bash
   pip install torch torchvision matplotlib scikit-learn
   ```

## Running the Code

### Training the Models
To train the models, navigate to the `src/` directory and run:
```bash
python train.py
```
This script will train both the custom CNN and EfficientNet models using the predefined parameters and save the trained models to the `models/` directory.

### Evaluating the Models
After training, evaluate the models by running:
```bash
python evaluate.py
```
This will load the trained models from the `models/` directory, perform evaluations on the test set, and save the output metrics and graphs in the `results/` directory.

## Function Descriptions
- `train_model()`: Function responsible for model training. It includes logging, saving checkpoints, and early stopping functionalities.
- `evaluate_model()`: Function to evaluate the model performance on the test dataset and generate a classification report and confusion matrix.
- `load_data()`: Function to load and preprocess the dataset. It also applies image augmentation techniques.

## Additional Notes
- Ensure the dataset path in `utils.py` is correctly set to the location where the Animals-10 dataset is stored.
- Model training and evaluation parameters can be adjusted in the `train.py` and `evaluate.py` scripts respectively.
