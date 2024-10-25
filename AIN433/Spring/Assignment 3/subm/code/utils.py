import logging
import os

import torch
from torchvision import datasets, transforms


def load_data(
    data_path, image_size=(128, 128), batch_size=64, split_ratio=(0.8, 0.1, 0.1)
):
    """
    Load and preprocess the dataset.

    Args:
        data_path (str): Path to the dataset directory.
        image_size (tuple): Tuple specifying the size to which the images will be resized.
        batch_size (int): Number of images in each batch.
        split_ratio (tuple): Tuple specifying the train, validation, and test split (should sum to 1).

    Returns:
        dict: Dictionary containing 'train', 'validate', and 'test' DataLoader objects.
    """
    # Define transformations
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load the dataset
    dataset = datasets.ImageFolder(root=data_path, transform=transform)

    # Calculate sizes of each data split
    total_size = len(dataset)
    train_size = int(total_size * split_ratio[0])
    val_size = int(total_size * split_ratio[1])
    test_size = total_size - train_size - val_size

    # Split the dataset
    train_dataset, validate_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Create data loaders
    data_loaders = {
        "train": torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        ),
        "validate": torch.utils.data.DataLoader(
            validate_dataset, batch_size=batch_size, shuffle=False
        ),
        "test": torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        ),
    }

    return data_loaders


def create_logger(filename, level=logging.INFO):
    """
    Create a logger for recording training progress.

    Args:
        filename (str): Filename to save the log.
        level (int): Logging level.

    Returns:
        Logger: Configured logger.
    """
    logging.basicConfig(
        filename=os.path.join("logs", filename),
        filemode="a",  # Append mode
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
    )
    logger = logging.getLogger()
    return logger
