import os
import sys

work_dir = "/Users/emre/GitHub/HU-AI/AIN432/Assignment 3"
# Add the path to your project root directory
if work_dir not in sys.path:
    sys.path.append(work_dir)

# Import your module
from kmeans import (  # noqa: E402
    elbow_method,
    visualize_clusters,
    visualize_superpixel_clusters,
)
from utils.create_dataset import (  # noqa: E402
    convert_save_df,
    excract_features,
    resize_images,
)
from utils.io import read_directory, read_images  # noqa: E402

# Directories
DATA_PATH = os.path.join(work_dir, "data")
PROCESSED_PATH = os.path.join(DATA_PATH, "processed")
INTERIM_PATH = os.path.join(DATA_PATH, "interim")
RAW_PATH = os.path.join(DATA_PATH, "raw")
MODELS_PATH = os.path.join(work_dir, "models")
GRAPH_PATH = os.path.join(work_dir, "graphs")
OUTPUTS_PATH = os.path.join(work_dir, "outputs")


def main():
    IMAGE_SIZE = (1024, 1024)

    # read images from directory
    image_paths = read_directory(RAW_PATH, file_extension=".jpg")
    images = read_images(RAW_PATH, image_paths)

    # resize images
    resized_images = resize_images(
        images, IMAGE_SIZE, save_imgs=True, save_dir=INTERIM_PATH, file_extension=".jpg"
    )

    # show resized images
    # show_images(resized_images, f"Resized Images ({IMAGE_SIZE})")

    df_pixel = excract_features(resized_images, feature_extractor="pixel_features")
    convert_save_df(
        df_pixel, ["R", "G", "B", "X", "Y"], PROCESSED_PATH, show_head=False
    )

    df_superpixel = excract_features(
        resized_images, feature_extractor="superpixel_features"
    )
    convert_save_df(
        df_superpixel,
        [
            "Mean R",
            "Mean G",
            "Mean B",
            "R Histogram",
            "G Histogram",
            "B Histogram",
            "Gabor 0.1 0",
            "Gabor 0.1 1",
            "Gabor 0.1 2",
            "Gabor 0.1 3",
            "Gabor 0.5 0",
            "Gabor 0.5 1",
            "Gabor 0.5 2",
            "Gabor 0.5 3",
        ],
        PROCESSED_PATH,
        prename="superpixel_",
        show_head=False,
    )

    for img_name in df_pixel:
        elbow_method(
            df_pixel[img_name],
            savefig=True,
            img_name=img_name.split(".")[0],
            save_dir=work_dir,
        )

    for img_name in df_pixel:
        for i in range(3, 11, 2):
            visualize_clusters(
                df_pixel[img_name],
                resized_images[img_name],
                k=i,
                savefig=True,
                img_name=img_name.split(".")[0],
                save_dir=work_dir,
            )

    for img_name in df_superpixel:
        elbow_method(
            df_superpixel[img_name],
            savefig=True,
            prename="_superpixel",
            img_name=img_name.split(".")[0],
            save_dir=work_dir,
        )

    for img_name in df_superpixel:
        for i in range(3, 11, 2):
            visualize_superpixel_clusters(
                df_superpixel[img_name],
                resized_images[img_name],
                k=i,
                savefig=True,
                img_name=img_name.split(".")[0],
                save_dir=work_dir,
                prename="_superpixel",
            )


if __name__ == "__main__":
    main()
