import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class GunsObjectDetectionDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_filenames = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_filename = self.image_filenames[idx]
        image_path = os.path.join(self.image_dir, image_filename)
        label_path = os.path.join(
            self.label_dir, image_filename.replace(".jpg", ".txt")
        )

        image = Image.open(image_path).convert("RGB")
        boxes = []
        labels = []

        with open(label_path, "r") as f:
            num_objects = int(f.readline().strip())
            for _ in range(num_objects):
                box = list(map(float, f.readline().strip().split()))
                x_min, y_min, x_max, y_max = box[0], box[1], box[2], box[3]
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(1)  # Assuming 1 is the class label for 'gun'

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}

        if self.transform:
            image = self.transform(image)

        return image, target


def show_samples(dataset, num_samples=4):
    fig, axs = plt.subplots(1, num_samples, figsize=(20, 5))
    axs = axs.flatten()

    for i in range(num_samples):
        image, target = dataset[i]
        image = np.array(image)

        axs[i].imshow(image)
        for box in target["boxes"]:
            x_min, y_min, x_max, y_max = box
            rect = patches.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
            axs[i].add_patch(rect)

        axs[i].axis("off")
    plt.show()


# Directory paths
image_dir = "/Users/emre/GitHub/HU-AI/AIN433/Spring/Assignment 4/dataset/Images"
label_dir = "/Users/emre/GitHub/HU-AI/AIN433/Spring/Assignment 4/dataset/Labels"

# Define your transformations (if any)
transform = None

# Create dataset
dataset = GunsObjectDetectionDataset(image_dir, label_dir, transform=transform)

# Create dataloader
dataloader = DataLoader(
    dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x))
)

# Show 4 samples
show_samples(dataset, num_samples=4)

# Example usage of dataloader
# for images, targets in dataloader:
#     print(images)
#     print(targets)
#     break
