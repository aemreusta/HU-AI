# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from skimage import color
from tqdm import tqdm
from utils.create_dataset import extract_superpixels
from utils.io import save_graphs


def kmeans(features, k, max_iters=100, tol=1e-4):
    # Randomly initialize centroids
    centroids = features[
        np.random.default_rng().choice(len(features), k, replace=False)
    ]

    for _ in range(max_iters):
        # Assign each feature to the nearest centroid
        distances = np.linalg.norm(features - centroids[:, np.newaxis], axis=2)
        labels = np.argmin(distances, axis=0)

        # Update centroids based on the mean of the assigned features
        new_centroids = np.array([features[labels == i].mean(axis=0) for i in range(k)])

        # Check convergence
        if np.linalg.norm(new_centroids - centroids) < tol:
            break

        centroids = new_centroids

    return labels, centroids


def elbow_method(
    features: np.ndarray,
    min_k: int = 3,
    max_k: int = 10,
    showfig=False,
    savefig=True,
    prename: str = "",
    img_name: str = "",
    save_dir: str = "",
):
    # print(features)

    inertias = []
    for k in tqdm(range(min_k, max_k + 1), desc=f"Running Elbow Method for {img_name}"):
        labels, centroids = kmeans(features, k)
        inertia = np.sum((features - centroids[labels]) ** 2)
        inertias.append(inertia)

    if showfig:
        plt.figure(figsize=(6, 4))
        plt.plot(range(min_k, max_k + 1), inertias, marker="o")
        plt.xlabel(f"Number of Clusters ({max_k})")
        plt.ylabel("Inertia")
        plt.title(f"Elbow Method for Optimal k at {img_name+prename}")
        plt.tight_layout()
        plt.show()

    if savefig:
        plt.figure(figsize=(6, 4))
        plt.plot(range(min_k, max_k + 1), inertias, marker="o")
        plt.xlabel(f"Number of Clusters ({max_k})")
        plt.ylabel("Inertia")
        plt.title(f"Elbow Method for Optimal k at {img_name+prename}")
        # plt.tight_layout()

        filename = img_name + f"{prename}_{max_k}_elbow_method.png"
        save_graphs(save_dir, "graphs", filename, plt)


def visualize_clusters(
    features: np.ndarray,
    image,
    k: int = 3,
    showfig=False,
    savefig=True,
    img_name: str = "",
    save_dir: str = "",
    prename: str = "",
):
    labels, _ = kmeans(features, k)

    cmap = ListedColormap(np.random.rand(len(np.unique(labels)), 3))
    labeled_image = color.label2rgb(
        labels.reshape(image.shape[:2]),
        image,
        kind="overlay",
        colors=cmap.colors,
        alpha=0.6,
    )

    if showfig:
        plt.figure(figsize=(8, 4))

        plt.subplot(121)
        plt.imshow(image)
        plt.title(f"Original Image {img_name+prename}")
        plt.axis("off")

        plt.subplot(122)
        plt.imshow(labeled_image)
        plt.title(f"{k} Cluster Segmented Image {img_name+prename}")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    if savefig:
        plt.figure(figsize=(8, 4))

        plt.subplot(121)
        plt.imshow(image)
        plt.title(f"Original Image {img_name+prename}")
        plt.axis("off")

        plt.subplot(122)
        plt.imshow(labeled_image)
        plt.title(f"{k} Cluster Segmented Image {img_name+prename}")
        plt.axis("off")

        # plt.tight_layout()

        filename = img_name + f"{prename}_{k}_segmented_image.png"
        save_graphs(save_dir, "graphs", filename, plt)


def visualize_superpixel_clusters(
    features: np.ndarray,
    image,
    k: int = 3,
    showfig=False,
    savefig=True,
    img_name: str = "",
    save_dir: str = "",
    prename: str = "superpixel_",
):
    labels, _ = kmeans(features, k)
    superpixels = extract_superpixels(image)

    # change superpixel values according to labels
    for idx, label in enumerate(labels):
        superpixels[superpixels == idx] = label

    cmap = ListedColormap(np.random.rand(len(np.unique(labels)), 3))
    labeled_image = color.label2rgb(
        superpixels,
        image,
        kind="overlay",
        colors=cmap.colors,
        alpha=0.6,
    )

    if showfig:
        plt.figure(figsize=(8, 4))

        plt.subplot(121)
        plt.imshow(image)
        plt.title(f"Original Image {img_name+prename}")
        plt.axis("off")

        plt.subplot(122)
        plt.imshow(labeled_image)
        plt.title(f"{k} Cluster Segmented Image {img_name+prename}")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    if savefig:
        plt.figure(figsize=(8, 4))

        plt.subplot(121)
        plt.imshow(image)
        plt.title(f"Original Image {img_name+prename}")
        plt.axis("off")

        plt.subplot(122)
        plt.imshow(labeled_image)
        plt.title(f"{k} Cluster Segmented Image {img_name+prename}")
        plt.axis("off")

        # plt.tight_layout()

        filename = img_name + f"{prename}_{k}_segmented_image.png"
        save_graphs(save_dir, "graphs", filename, plt)
