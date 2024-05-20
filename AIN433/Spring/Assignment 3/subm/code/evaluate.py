import matplotlib.pyplot as plt
import torch
from model import CustomCNN  # Import other models as needed
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Constants
IMAGE_SIZE = (128, 128)
DATASET_PATH = "path_to_your_dataset/test"  # Specify the path to your test dataset
MODEL_PATH = "models/model_epoch_9.pth"  # Adjust the path as necessary

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transformations
transform = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Load the dataset
test_dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)


# Function to evaluate the model
def evaluate_model(model, device, test_loader):
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(device)
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Calculate metrics
    print("Classification Report:")
    print(
        classification_report(all_labels, all_preds, target_names=test_dataset.classes)
    )

    # Calculate and display the confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=test_dataset.classes
    )

    plt.figure(figsize=(10, 10))
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()


# Main execution
if __name__ == "__main__":
    # Select and load your model
    model = CustomCNN(num_classes=len(test_dataset.classes))  # Or load another model
    evaluate_model(model, device, test_loader)
