import logging
import os

import torch
import torch.optim as optim
from model import CustomCNN
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.notebook import tqdm

# Set device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device.type.upper()} for computation.")

# Constants
BATCH_SIZE = 64
IMAGE_SIZE = (128, 128)
DATASET_PATH = "path_to_your_dataset"

# Create necessary directories
os.makedirs("graphs", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Dataset loading and transformations
data_transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

dataset = datasets.ImageFolder(DATASET_PATH, transform=data_transforms)

# Splitting dataset into train, validation, and test sets
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size, test_size]
)

# DataLoader setup
train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
)
val_dataloader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
)
test_dataloader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
)


# Training function
def train_model(model, dataloaders, num_epochs, save_path, logger, learning_rate=0.001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for images, labels in tqdm(dataloaders["train"]):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloaders["train"])
        logger.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # Save model checkpoint
        torch.save(
            model.state_dict(), os.path.join("models", save_path.format(epoch=epoch))
        )

    print("Training complete.")


def create_logger(filename):
    logging.basicConfig(
        filename=os.path.join("logs", filename),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger()


# Main function to execute training
if __name__ == "__main__":
    logger = create_logger("training.log")
    model = CustomCNN(num_classes=10)  # Change this to switch between models
    dataloaders = {
        "train": train_dataloader,
        "validate": val_dataloader,
        "test": test_dataloader,
    }
    train_model(
        model,
        dataloaders,
        num_epochs=10,
        save_path="model_epoch_{epoch}.pth",
        logger=logger,
    )
