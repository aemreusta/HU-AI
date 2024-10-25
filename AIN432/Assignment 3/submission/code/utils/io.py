import os

import cv2
import matplotlib.pyplot as plt


def read_directory(directory_name, file_extension=".png"):
    return sorted(
        [file for file in os.listdir(directory_name) if file.endswith(file_extension)]
    )


def create_directory(workdir, directory_name):
    if not os.path.exists(os.path.join(workdir, directory_name)):
        os.makedirs(os.path.join(workdir, directory_name))


def save_graphs(workdir, directory_name, graphname, plt):
    filename = os.path.join(workdir, directory_name, graphname)

    if not os.path.isfile(filename):
        create_directory(workdir, directory_name)
        plt.savefig(filename)
        print("Graph succesfully saved: ", filename)
        plt.close()

    else:
        print("Graph already exist: ", filename)


def read_images(work_dir, image_paths):
    images = {}
    for image_path in image_paths:
        # check if image exist
        if not os.path.isfile(os.path.join(work_dir, image_path)):
            print("Image not found: ", image_path)
            continue

        # read image
        img = cv2.imread(os.path.join(work_dir, image_path))
        # cobvert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images[image_path] = img

    return images


def show_images(images, title="Images", work_dir=None):
    images_len = len(images)
    cols = 3
    rows = images_len // cols + (images_len % cols > 0)

    fig, ax = plt.subplots(rows, cols, figsize=(9, 3 * rows))
    fig.suptitle(title)
    fig.tight_layout()

    axes = ax.ravel() if images_len > cols else [ax]

    if type(images) is dict:  # noqa: E721
        for i, (image_name, image) in enumerate(images.items()):
            axes[i].imshow(image, aspect="auto")
            axes[i].set_title(image_name, fontsize=8)
            # axes[i].axis("off")

    else:
        for i, image in enumerate(images):
            axes[i].imshow(image, aspect="auto")
            axes[i].axis("off")

    # Hide any unused subplots
    for j in range(i + 1, rows * cols):
        axes[j].axis("off")

    save_graphs(work_dir, "graphs", title + ".png", plt)
    plt.show()
