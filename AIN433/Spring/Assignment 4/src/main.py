import logging
import os
import time
from collections import defaultdict
from datetime import datetime

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as torchmodels
from PIL import Image
from pytorchyolo import models
from torch.utils.data import DataLoader, Dataset, random_split
from torchsummary import summary
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def create_logger(filename):
    logging.basicConfig(
        filename=os.path.join("logs", filename),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger()


class GunsObjectDetectionDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, image_size=(224, 224)):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_size = image_size
        self.image_filenames = [f for f in os.listdir(image_dir) if f.endswith(".jpeg")]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_filename = self.image_filenames[idx]
        image_path = os.path.join(self.image_dir, image_filename)
        label_path = os.path.join(
            self.label_dir, image_filename.replace(".jpeg", ".txt")
        )

        image = Image.open(image_path).convert("RGB")
        original_size = image.size

        boxes = []
        labels = []

        with open(label_path, "r") as f:
            num_objects = int(f.readline().strip())
            for _ in range(num_objects):
                box = list(map(float, f.readline().strip().split()))
                x_min, y_min, x_max, y_max = box[0], box[1], box[2], box[3]
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(1)  # Assuming 1 is the class label for 'gun'

        # If there are no objects, set labels to 0 (background)
        if len(boxes) == 0:
            labels.append(0)
            boxes.append([0, 0, 0, 0])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        if self.transform:
            image, boxes = self.apply_transforms(image, boxes, original_size)

        target = {"boxes": boxes, "labels": labels}

        return image, target

    def apply_transforms(self, image, boxes, original_size):
        if isinstance(self.transform, transforms.Compose):
            for t in self.transform.transforms:
                if isinstance(t, transforms.Resize):
                    image = t(image)
                    resized_size = image.size
                    scale_x = resized_size[0] / original_size[0]
                    scale_y = resized_size[1] / original_size[1]
                    boxes[:, [0, 2]] *= scale_x
                    boxes[:, [1, 3]] *= scale_y
                elif isinstance(t, transforms.RandomHorizontalFlip):
                    if torch.rand(1) < 0.5:
                        image = t(image)
                        width = image.size[0]
                        boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                elif isinstance(t, transforms.RandomCrop):
                    i, j, h, w = transforms.RandomCrop.get_params(
                        image, output_size=self.image_size
                    )
                    image = transforms.functional.crop(image, i, j, h, w)
                    boxes[:, [0, 2]] -= j
                    boxes[:, [1, 3]] -= i
                    boxes[:, 0] = boxes[:, 0].clamp(min=0, max=w)
                    boxes[:, 2] = boxes[:, 2].clamp(min=0, max=w)
                    boxes[:, 1] = boxes[:, 1].clamp(min=0, max=h)
                    boxes[:, 3] = boxes[:, 3].clamp(min=0, max=h)
                elif isinstance(t, transforms.ColorJitter):
                    image = t(image)
                elif isinstance(t, transforms.ToTensor):
                    image = t(image)
                elif isinstance(t, transforms.Normalize):
                    image = t(image)
        else:
            image = self.transform(image)

        return image, boxes


def show_samples(dataset, logger, num_samples=8, max_per_row=4):
    num_rows = (num_samples + max_per_row - 1) // max_per_row
    fig, axs = plt.subplots(
        num_rows, max_per_row, figsize=(max_per_row * 5, num_rows * 5)
    )
    axs = axs.flatten()

    # Mean and std used for normalization in the dataset
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Get random indices using NumPy's random generator
    random_indices = np.random.choice(len(dataset), num_samples, replace=False)

    for i, idx in enumerate(random_indices):
        image, target = dataset[idx]
        image = np.array(image).transpose(1, 2, 0)  # Transpose the image

        # Unnormalize the image
        image = (image * std) + mean
        image = np.clip(image, 0, 1)  # Clip to [0, 1] range

        axs[i].imshow(image)
        for box, label in zip(target["boxes"], target["labels"]):
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
            axs[i].text(
                x_min,
                y_min,
                f"Class {label}",
                color="white",
                fontsize=8,
                bbox=dict(facecolor="red", alpha=0.5),
            )

        axs[i].axis("off")

    # Hide any unused subplots
    for j in range(i + 1, len(axs)):
        axs[j].axis("off")

    fig.suptitle("Dataset Samples", fontsize=16, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join("graphs", "dataset_samples.png"))
    plt.close()

    # Print a message to indicate that the samples are shown
    logger.info("Samples are shown and saved to 'graphs' directory.")


def plot_split_sizes(train_size, val_size, test_size, logger):
    # Create the plot
    plt.figure(figsize=(8, 6))
    bars = plt.bar(["Train", "Validation", "Test"], [train_size, val_size, test_size])

    # Annotate each bar with the number of samples
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
        )

    # Set labels and title
    plt.xlabel("Split")
    plt.ylabel("Number of samples")
    plt.title("Dataset split")

    plt.tight_layout()
    # Save the plot
    plt.savefig(os.path.join("graphs", "dataset_split.png"))
    plt.close()

    # Print a message to indicate that the plot is saved
    logger.info("Dataset split plot is saved to 'graphs' directory.")


def compute_loss(
    cls_out, reg_out, cls_targets, reg_targets, criterion_cls, criterion_reg
):
    cls_loss = criterion_cls(cls_out, cls_targets)
    reg_loss = criterion_reg(reg_out, reg_targets)
    loss = cls_loss + reg_loss
    return loss, cls_loss, reg_loss


def compute_iou(pred_boxes, true_boxes):
    # Intersection over Union (IoU) calculation
    inter_xmin = torch.max(pred_boxes[:, 0], true_boxes[:, 0])
    inter_ymin = torch.max(pred_boxes[:, 1], true_boxes[:, 1])
    inter_xmax = torch.min(pred_boxes[:, 2], true_boxes[:, 2])
    inter_ymax = torch.min(pred_boxes[:, 3], true_boxes[:, 3])

    inter_area = torch.clamp(inter_xmax - inter_xmin, min=0) * torch.clamp(
        inter_ymax - inter_ymin, min=0
    )
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (
        pred_boxes[:, 3] - pred_boxes[:, 1]
    )
    true_area = (true_boxes[:, 2] - true_boxes[:, 0]) * (
        true_boxes[:, 3] - true_boxes[:, 1]
    )

    union_area = pred_area + true_area - inter_area
    iou = inter_area / union_area

    return iou.mean().item()


def compute_iou_np(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def non_max_suppression(boxes, scores, iou_threshold=0.5):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def sliding_window(image, window_size, stride):
    """Generate sliding windows for the given image."""
    windows = []
    _, height, width = image.shape

    for y in range(0, height - window_size + 1, stride):
        for x in range(0, width - window_size + 1, stride):
            window = image[:, y : y + window_size, x : x + window_size]
            windows.append((window, (x, y)))

    return windows


class OverfeatWithEfficientNet(nn.Module):
    def __init__(self, debug=False):
        super(OverfeatWithEfficientNet, self).__init__()
        self.debug = debug

        # Load the pre-trained EfficientNetB0
        efficientnet = torchmodels.efficientnet_b0(
            weights=EfficientNet_B0_Weights.DEFAULT
        )

        # Freeze the parameters of the pre-trained layers
        for param in efficientnet.parameters():
            param.requires_grad = False

        # Use the EfficientNet up to the last convolutional layer
        self.backbone = nn.Sequential(*list(efficientnet.children())[:-1])

        # Flatten the output of the backbone
        self.flatten = nn.Flatten()

        # Classification head with more layers
        self.classifier = nn.Sequential(
            nn.Linear(1280, 512),  # Updated input size
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),  # 2 classes: background and gun
        )

        # Regression head for bounding box with more layers
        self.regressor = nn.Sequential(
            nn.Linear(1280, 512),  # Updated input size
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4),  # 4 coordinates: x_min, y_min, x_max, y_max
        )

    def forward(self, x):
        if self.debug:
            print("Input shape:", x.shape)

        features = self.backbone(x)

        if self.debug:
            print("Features shape after backbone:", features.shape)

        features = self.flatten(features)

        if self.debug:
            print("Features shape after flattening:", features.shape)

        cls_out = self.classifier(features)

        if self.debug:
            print("Classification output shape:", cls_out.shape)

        reg_out = self.regressor(features)

        if self.debug:
            print("Regression output shape:", reg_out.shape)

        return cls_out, reg_out


class YOLOv3TinyModel(nn.Module):
    def __init__(self, num_classes=2, debug=False):
        super(YOLOv3TinyModel, self).__init__()
        self.debug = debug

        # Load the pre-trained YOLOv3-tiny model using pytorchyolo
        yolo_v3_tiny = models.load_model(
            r"C:\Users\aliseydi\Git\Assignment 4\models\pretrained_weights\PyTorch-YOLOv3/config/yolov3-tiny.cfg",
            r"C:\Users\aliseydi\Git\Assignment 4\models\pretrained_weights\PyTorch-YOLOv3/weights/yolov3-tiny.weights",
        )

        # Freeze all layers except the last three layers
        for param in list(yolo_v3_tiny.parameters())[:-3]:
            param.requires_grad = False

        # Modify the last layer to accommodate the number of classes required
        yolo_v3_tiny.module_list[-1] = nn.Conv2d(
            in_channels=255, out_channels=num_classes + 5, kernel_size=1
        )

        self.backbone = nn.Sequential(*list(yolo_v3_tiny.module_list[:-1]))
        self.head = yolo_v3_tiny.module_list[-1]

        self.flatten = nn.Flatten()

        # Classification head with more layers
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

        # Regression head for bounding box with more layers
        self.regressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4),  # 4 coordinates: x_min, y_min, x_max, y_max
        )

    def forward(self, x):
        if self.debug:
            print("Input shape:", x.shape)

        features = self.backbone(x)

        if self.debug:
            print("Features shape after backbone:", features.shape)

        features = self.flatten(features)

        if self.debug:
            print("Features shape after flattening:", features.shape)

        cls_out = self.classifier(features)

        if self.debug:
            print("Classification output shape:", cls_out.shape)

        reg_out = self.regressor(features)

        if self.debug:
            print("Regression output shape:", reg_out.shape)

        return cls_out, reg_out


def train_model(
    model,
    train_loader,
    val_loader,
    epochs=10,
    learning_rate=1e-4,
    device="cuda",
    logger=None,
    debug=False,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion_cls = nn.CrossEntropyLoss().to(device)
    criterion_reg = nn.MSELoss().to(device)

    history = defaultdict(list)
    best_loss = float("inf")
    best_model_wts = model.state_dict()
    start_time = time.time()

    def compute_metrics(loader, is_train=True):
        model.train() if is_train else model.eval()
        total_loss, total_cls_loss, total_reg_loss = 0, 0, 0
        total_correct, total_cls_samples, total_iou = 0, 0, 0

        for images, targets in loader:
            images = torch.stack(images).to(device)
            cls_targets = torch.cat([t["labels"] for t in targets], dim=0).to(device)
            reg_targets = torch.cat([t["boxes"] for t in targets], dim=0).to(device)

            if is_train:
                optimizer.zero_grad()
                cls_out, reg_out = model(images)
                cls_targets = cls_targets[: cls_out.size(0)]
                reg_targets = reg_targets[: reg_out.size(0)]
                total_loss, cls_loss, reg_loss = compute_loss(
                    cls_out,
                    reg_out,
                    cls_targets,
                    reg_targets,
                    criterion_cls,
                    criterion_reg,
                )
                total_loss.backward()
                optimizer.step()
                # convert total loss tensor to numpy cpu
                total_loss = total_loss.cpu().detach().numpy()

            else:
                with torch.no_grad():
                    cls_out, reg_out = model(images)
                    cls_targets = cls_targets[: cls_out.size(0)]
                    reg_targets = reg_targets[: reg_out.size(0)]
                    total_loss, cls_loss, reg_loss = compute_loss(
                        cls_out,
                        reg_out,
                        cls_targets,
                        reg_targets,
                        criterion_cls,
                        criterion_reg,
                    )

            total_loss += total_loss.item()
            total_cls_loss += cls_loss.item()
            total_reg_loss += reg_loss.item()
            _, preds = torch.max(cls_out, 1)
            total_correct += (preds == cls_targets).sum().item()
            total_cls_samples += cls_targets.size(0)
            total_iou += compute_iou(reg_out, reg_targets)

        avg_loss = total_loss / len(loader)
        avg_cls_loss = total_cls_loss / len(loader)
        avg_reg_loss = total_reg_loss / len(loader)
        avg_accuracy = total_correct / total_cls_samples
        avg_iou = total_iou / len(loader)

        if debug:
            print(
                type(avg_loss),
                type(avg_cls_loss),
                type(avg_reg_loss),
                type(avg_accuracy),
                type(avg_iou),
            )

        return (
            avg_loss,
            avg_cls_loss,
            avg_reg_loss,
            avg_accuracy,
            avg_iou,
        )

    # use tqdm for progress bar
    for epoch in tqdm(range(epochs), desc="Epochs", unit="epoch"):
        train_metrics = compute_metrics(train_loader, is_train=True)
        val_metrics = compute_metrics(val_loader, is_train=False)

        # Training metrics
        history["train_loss"].append(train_metrics[0])
        history["train_cls_loss"].append(train_metrics[1])
        history["train_reg_loss"].append(train_metrics[2])
        history["train_accuracy"].append(train_metrics[3])
        history["train_iou"].append(train_metrics[4])

        # Validation metrics
        history["val_loss"].append(val_metrics[0])
        history["val_cls_loss"].append(val_metrics[1])
        history["val_reg_loss"].append(val_metrics[2])
        history["val_accuracy"].append(val_metrics[3])
        history["val_iou"].append(val_metrics[4])

        if val_metrics[0] < best_loss:
            best_loss = val_metrics[0]
            best_model_wts = model.state_dict()
            torch.save(best_model_wts, os.path.join("models", "best_model.pth"))

        epoch_time = time.time() - start_time
        logger.info(
            f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_metrics[0]:.4f}, Val Loss: {val_metrics[0]:.4f}, "
            f"Train CLS Loss: {train_metrics[1]:.4f}, Val CLS Loss: {val_metrics[1]:.4f}, Train REG Loss: {train_metrics[2]:.4f}, "
            f"Val REG Loss: {val_metrics[2]:.4f}, Train Accuracy: {train_metrics[3]:.4f}, Val Accuracy: {val_metrics[3]:.4f}, "
            f"Train mIoU: {train_metrics[4]:.4f}, Val mIoU: {val_metrics[4]:.4f}, Time: {epoch_time:.2f}s"
        )

    # close tqdm with message
    tqdm.write("Training completed.")

    model.load_state_dict(best_model_wts)
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time // 60:.0f}m {total_time % 60:.0f}s")
    logger.info(f"Best validation loss: {best_loss:.4f}")

    return model, history


def plot_history(history, logger=None, debug=False):
    if debug:
        for k, v in history.items():
            print(k, type(v), len(v))
            # print random 5 elements
            if isinstance(v, list):
                print(v[:5])

    # Convert CUDA tensors to CPU tensors and then to numpy if necessary
    for k, v in history.items():
        if isinstance(v, torch.Tensor):
            history[k] = v.cpu().numpy()
        elif isinstance(v, list) and isinstance(v[0], torch.Tensor):
            history[k] = [t.cpu().numpy() for t in v]

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(18, 12))

    # Plot Training and Validation Loss
    plt.subplot(2, 3, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss", color="blue")
    plt.plot(epochs, history["val_loss"], label="Val Loss", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    # Plot Training and Validation Classification Loss
    plt.subplot(2, 3, 2)
    plt.plot(epochs, history["train_cls_loss"], label="Train CLS Loss", color="blue")
    plt.plot(epochs, history["val_cls_loss"], label="Val CLS Loss", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("CLS Loss")
    plt.title("Training and Validation Classification Loss")
    plt.legend()

    # Plot Training and Validation Regression Loss
    plt.subplot(2, 3, 3)
    plt.plot(epochs, history["train_reg_loss"], label="Train REG Loss", color="blue")
    plt.plot(epochs, history["val_reg_loss"], label="Val REG Loss", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("REG Loss")
    plt.title("Training and Validation Regression Loss")
    plt.legend()

    # Plot Training and Validation Accuracy
    plt.subplot(2, 3, 4)
    plt.plot(epochs, history["train_accuracy"], label="Train Accuracy", color="blue")
    plt.plot(epochs, history["val_accuracy"], label="Val Accuracy", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()

    # Plot Training and Validation mIoU
    plt.subplot(2, 3, 5)
    plt.plot(epochs, history["train_iou"], label="Train mIoU", color="blue")
    plt.plot(epochs, history["val_iou"], label="Val mIoU", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("mIoU")
    plt.title("Training and Validation mIoU")
    plt.legend()

    # Add bold and big title
    plt.suptitle("Training History", fontsize=16, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join("graphs", "training_history.png"))
    plt.close()

    if logger:
        logger.info("Training history plot is saved to 'graphs' directory.")


def evaluate_model(model, dataloader, window_size, stride, iou_threshold, device):
    """Evaluate the model using sliding window and non-max suppression."""
    model.eval()
    total_iou = 0
    total_correct = 0
    total_samples = 0
    all_predictions = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = torch.stack(images).to(device)
            batch_predictions = []

            for image, target in zip(images, targets):
                windows = sliding_window(image, window_size, stride)
                boxes = []
                scores = []

                for window, (x, y) in windows:
                    window = window.unsqueeze(0).to(device)
                    outputs = model(window)

                    # Classification and Regression outputs
                    class_scores = F.softmax(outputs[0], dim=1)
                    bbox_preds = outputs[1]

                    max_score, predicted_class = torch.max(class_scores, dim=1)

                    if max_score.item() > 0.5:  # Threshold score
                        boxes.append(
                            [
                                x + bbox_preds[0][0].item(),
                                y + bbox_preds[0][1].item(),
                                x + bbox_preds[0][2].item(),
                                y + bbox_preds[0][3].item(),
                            ]
                        )
                        scores.append(max_score.item())

                if boxes:
                    keep = non_max_suppression(boxes, scores, iou_threshold)
                    boxes = [boxes[i] for i in keep]
                    scores = [scores[i] for i in keep]

                    batch_predictions.append({"boxes": boxes, "scores": scores})

                    # Calculate mIoU
                    gt_boxes = target["boxes"].cpu().numpy()
                    if len(boxes) > 0 and len(gt_boxes) > 0:
                        ious = [
                            compute_iou_np(pred_box, gt_box)
                            for pred_box in boxes
                            for gt_box in gt_boxes
                        ]
                        total_iou += sum(ious) / len(ious)
                        total_correct += sum(1 for iou in ious if iou > 0.5)
                        total_samples += len(gt_boxes)

            all_predictions.append(batch_predictions)

    mIoU = total_iou / total_samples if total_samples > 0 else 0
    accuracy = total_correct / total_samples if total_samples > 0 else 0

    return mIoU, accuracy, all_predictions


def visualize_evaluation_results(
    test_loader,
    predictions,
    mIoU,
    accuracy,
    max_per_row=4,
    logger=None,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
):
    """
    Visualize the evaluation results and display mIoU and accuracy.

    Parameters:
    - test_loader: DataLoader containing the test dataset.
    - predictions: List of predictions containing boxes and scores.
    - mIoU: Mean Intersection over Union.
    - accuracy: Classification accuracy.
    - max_per_row: Maximum number of images to display in a row.
    - mean: Mean used for normalization.
    - std: Standard deviation used for normalization.
    """
    images, targets = [], []
    for images_batch, targets_batch in test_loader:
        images.extend(images_batch)
        targets.extend(targets_batch)

    # extend predictions too
    predictions = [pred for batch in predictions for pred in batch]

    num_samples = len(images)
    num_rows = (num_samples + max_per_row - 1) // max_per_row

    fig, axs = plt.subplots(
        num_rows, max_per_row, figsize=(max_per_row * 5, num_rows * 5)
    )
    axs = axs.flatten()

    # print(len(targets), len(predictions))

    for i, (image, target, prediction) in enumerate(zip(images, targets, predictions)):
        if i >= len(axs):
            break
        ax = axs[i]

        # Unnormalize the image
        image_np = image.permute(1, 2, 0).cpu().numpy()
        image_np = image_np * std + mean
        image_np = (image_np * 255).astype("uint8")

        ax.imshow(image_np)

        # Plot ground truth boxes
        for box in target["boxes"].cpu().numpy():
            x1, y1, x2, y2 = box
            width, height = x2 - x1, y2 - y1
            rect = patches.Rectangle(
                (x1, y1),
                width,
                height,
                linewidth=2,
                edgecolor="green",
                facecolor="none",
            )
            ax.add_patch(rect)
            ax.text(
                x1,
                y1 - 10,
                "Ground Truth",
                color="green",
                fontsize=9,
                fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.5),
            )

        for box, score in zip(prediction["boxes"], prediction["scores"]):
            x1, y1, x2, y2 = box
            width, height = x2 - x1, y2 - y1
            rect = patches.Rectangle(
                (x1, y1),
                width,
                height,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )
            ax.add_patch(rect)
            ax.text(
                x1,
                y1 - 30,
                f"{score:.2f}",
                color="red",
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.5),
            )

        ax.set_title(f"Evaluation Result {i + 1}")
        ax.axis("off")

    # Hide any unused subplots
    for j in range(i + 1, len(axs)):
        axs[j].axis("off")

    # Display mIoU and accuracy
    fig.suptitle(
        f"Evaluation Results\nmIoU: {mIoU:.4f}, Accuracy: {accuracy:.4f}",
        fontsize=16,
        fontweight="bold",
    )

    if not os.path.exists("graphs"):
        os.makedirs("graphs")

    plt.savefig(os.path.join("graphs", "evaluation_results.png"))

    if logger:
        logger.info("Evaluation results are saved to 'graphs' directory.")

    plt.close()


def visualize_metrics(mIoU, accuracy):
    """
    Visualizes the mIoU and accuracy metrics using bar plots.

    Parameters:
    mIoU (float): Mean Intersection over Union.
    accuracy (float): Accuracy of the model.
    """

    # Data for plotting
    metrics = ["Mean IoU", "Accuracy"]
    values = [mIoU, accuracy]

    # Create bar plot
    plt.figure(figsize=(8, 6))
    plt.bar(metrics, values, color=["blue", "green"])

    # Add title and labels
    plt.title("Model Evaluation Metrics")
    plt.xlabel("Metrics")
    plt.ylabel("Values")

    # Add value labels on top of bars
    for i in range(len(metrics)):
        plt.text(i, values[i] + 0.01, f"{values[i]:.4f}", ha="center", va="bottom")

    # Display the plot
    plt.ylim(0, 1)

    # add bold and big titl
    plt.suptitle("Model Evaluation Metrics", fontsize=16, fontweight="bold")

    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join("graphs", "evaluation_metrics.png"))
    plt.close()


def main(debug=False):
    # Create necessary directories
    os.makedirs("graphs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    log_filename = f"log_train_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    logger = create_logger(log_filename)

    # Print the logger filename
    print(f"Logging to {log_filename}")
    logger.info("Starting the training process.")

    # log created files
    logger.info("Created directories: 'graphs', 'models', 'logs'")

    # Directory paths
    DATASET_DIR = r"C:\Users\aliseydi\Git\Assignment 4\dataset"
    image_dir = os.path.join(DATASET_DIR, "Images")
    label_dir = os.path.join(DATASET_DIR, "Labels")

    # print the dataset directory
    logger.info(f"Images readed from {image_dir}, Labels readed from {label_dir}")

    # Set random seed for reproducibility
    MANUAL_SEED = 42
    torch.manual_seed(MANUAL_SEED)

    # log manual seed
    logger.info(f"Set manual seed {MANUAL_SEED} for reproducibility.")

    BATCH_SIZE = 8
    EPOCHS = 200
    IMAGE_SIZE = (128, 128)

    # print the default batch size and epochs
    logger.info(
        f"Batch size: {BATCH_SIZE}, Epochs: {EPOCHS}, Image size: {IMAGE_SIZE[0]} x {IMAGE_SIZE[1]}"
    )

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print the device
    logger.info(f"Using {device.type.upper()} for computation.")

    # Improved data transforms for object detection
    data_transforms = transforms.Compose(
        [
            transforms.Resize(
                (int(IMAGE_SIZE[0] * 1.2), int(IMAGE_SIZE[1] * 1.2))
            ),  # Resize to a larger size
            transforms.RandomCrop(IMAGE_SIZE),  # Randomly crop to the desired size
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
            ),  # Random color jitter
            # transforms.RandomHorizontalFlip(),  # Randomly flip horizontally
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # Normalize
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),  # Resize to the desired size
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # Normalize
        ]
    )

    # Create dataset
    dataset = GunsObjectDetectionDataset(
        image_dir, label_dir, transform=data_transforms, image_size=IMAGE_SIZE
    )

    # Check if dataset is correctly loaded
    logger.info(f"Dataset length: {len(dataset)}")

    if len(dataset) == 0:
        raise ValueError("No data found in the dataset. Please check the dataset path.")

    # Show 4 samples
    show_samples(dataset, num_samples=12, logger=logger)

    # Calculate lengths for train, validation, and test splits
    total_size = len(dataset)
    train_size = int(0.6 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size

    # print the sizes of the splits
    logger.info(
        f"Train size: {train_size}, Validation size: {val_size}, Test size: {test_size}"
    )

    if not os.path.exists(os.path.join("graphs", "dataset_split.png")):
        # Plot the split sizes
        plot_split_sizes(train_size, val_size, test_size, logger=logger)

    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Apply test transforms to the test dataset
    test_dataset.dataset.transform = test_transforms

    # Define model configurations
    model_configs = [
        ("YOLOv3Tiny", YOLOv3TinyModel, [8, 16], [1e-4, 1e-6]),
        ("EfficientNet", OverfeatWithEfficientNet, [8, 16], [1e-4, 1e-6]),
    ]

    for model_name, model_class, batch_sizes, learning_rates in model_configs:
        for batch_size in batch_sizes:
            for lr in learning_rates:
                # Create DataLoaders
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    collate_fn=lambda x: tuple(zip(*x)),
                )
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=lambda x: tuple(zip(*x)),
                )
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=lambda x: tuple(zip(*x)),
                )

                # Log batch size and learning rate
                logger.info(
                    f"Training {model_name} with batch size: {batch_size}, learning rate: {lr}"
                )

                model = model_class(debug=debug).to(device)

                if debug:
                    summary(model, (3, IMAGE_SIZE[0], IMAGE_SIZE[1]))

                # Train the model
                best_model, history = train_model(
                    model,
                    train_loader,
                    val_loader,
                    epochs=200,
                    learning_rate=lr,
                    device=device,
                    logger=logger,
                    debug=debug,
                )

                # Plot the training history
                plot_history(history, logger, debug=debug)

                window_size = 64  # Example window size
                stride = 16  # Example stride
                iou_threshold = 0.5  # Example IoU threshold

                mIoU, accuracy, predictions = evaluate_model(
                    best_model, test_loader, window_size, stride, iou_threshold, device
                )

                logger.info(
                    f"{model_name} - Mean IoU: {mIoU:.4f}, Accuracy: {accuracy:.4f}"
                )

                # Visualize the evaluation metrics
                visualize_metrics(mIoU, accuracy)

                # Visualize the evaluation results
                visualize_evaluation_results(
                    test_loader, predictions, mIoU, accuracy, logger=logger
                )

                # Save the best model
                model_save_path = os.path.join(
                    "models", f"{model_name}_bs{batch_size}_lr{lr}.pth"
                )
                torch.save(best_model.state_dict(), model_save_path)
                logger.info(f"Saved {model_name} model to {model_save_path}")


if __name__ == "__main__":
    main(debug=False)
