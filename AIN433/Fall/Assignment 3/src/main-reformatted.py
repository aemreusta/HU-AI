import os
import random

import cv2
import numpy as np
from matplotlib import pyplot as plt

working_dir = "/Users/emre/GitHub/HU-AI/AIN433/Assignment 3"
DATASETS_PATH = os.path.join(working_dir, "datasets")
RAW_DATASET_PATH = os.path.join(DATASETS_PATH, "raw")
OUTPUTS_PATH = os.path.join(working_dir, "ouputs")

# SURF = cv2.xfeatures2D.SURF_create()

NEAREST_NEIGHBOR_NUM = 2
RANSAC_THRESH = 2
INLIER_THRESH = 2
RANSAC_ITERATION = 1000

index_params = dict(algorithm=0, trees=5)
search_params = dict(checks=100)
# Matchers
brute_force_matcher = cv2.BFMatcher()
flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)


def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    # else:
    # print(f"Directory '{directory_path}' already exists.")


def get_subset_names(directory_path):
    subsets = {}

    for root, directories, files in os.walk(directory_path):
        for directory in sorted(directories):
            subset_path = os.path.join(root, directory)
            image_paths = [
                os.path.join(subset_path, file)
                for file in sorted(os.listdir(subset_path))
            ]
            subsets[directory] = image_paths

    return subsets


def feature_extraction(img, img_gray, feature_points_plot, extractor):
    # Feature extraction: find the key points, compute the descriptors with SURF
    key_pts, descs = extractor.detectAndCompute(img_gray, None)
    # Plots showing feature points for each ordered pair of sub-image
    drawn_key_pts = cv2.drawKeypoints(
        img,
        key_pts,
        None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    if feature_points_plot is None:
        # Assign first image's feature points
        feature_points_plot = drawn_key_pts
    else:
        # Merge first and second images' feature points
        feature_points_plot = np.concatenate(
            (feature_points_plot, drawn_key_pts), axis=1
        )
        # Plot feature points of images
        plt.imshow(cv2.cvtColor(feature_points_plot, cv2.COLOR_BGR2RGB))
        plt.title("Feature Points"), plt.xticks([]), plt.yticks([])
        # plt.show()

    return key_pts, descs, feature_points_plot


def feature_matching(
    cur_image, cur_feature_pts, cur_descs, next_image, next_feature_pts, next_descs
):
    if (
        cur_descs is not None
        and len(cur_descs) > NEAREST_NEIGHBOR_NUM
        and next_descs is not None
        and len(next_descs) > NEAREST_NEIGHBOR_NUM
    ):
        # Get knn detector
        matches = flann_matcher.knnMatch(next_descs, cur_descs, k=NEAREST_NEIGHBOR_NUM)

        # Apply ratio test
        good_matches = []

        for i, pair in enumerate(matches):
            try:
                m, n = pair
                if m.distance < 0.6 * n.distance:
                    good_matches.append([m])

            except ValueError:
                return None

        matches = np.asarray(good_matches)

        # cv2.drawMatchesKnn expects list of lists as matches.
        drawn_matches = cv2.drawMatchesKnn(
            cur_image,
            cur_feature_pts,
            next_image,
            next_feature_pts,
            good_matches,
            None,
            flags=2,
        )
        plt.imshow(cv2.cvtColor(drawn_matches, cv2.COLOR_BGR2RGB))
        plt.title("Feature Point Matching Lines"), plt.xticks([]), plt.yticks([])
        plt.show()
        return matches

    else:
        plt.imshow(
            cv2.cvtColor(
                np.concatenate((cur_image, next_image), axis=1), cv2.COLOR_BGR2RGB
            )
        )
        plt.title("Feature Point Matching Lines"), plt.xticks([]), plt.yticks([])
        plt.show()
        return None


def RANSAC(match_pairs):
    max_inliers = 0
    best_H = None

    for i in range(RANSAC_ITERATION):
        # Get 4 random matches
        random_matches = np.vstack(
            (
                match_pairs[random.randrange(0, len(match_pairs))],
                match_pairs[random.randrange(0, len(match_pairs))],
                match_pairs[random.randrange(0, len(match_pairs))],
                match_pairs[random.randrange(0, len(match_pairs))],
            )
        )

        # Get homography matrix
        H = find_homography(random_matches)

        # Compute inliers where ||pi’, H pi || <ε
        cur_inliers = 0
        for i in range(len(match_pairs)):
            geo_dist = get_geometric_distance(match_pairs[i], H)
            if geo_dist < INLIER_THRESH:
                cur_inliers += 1

        # Update maximum inlier number
        if cur_inliers > max_inliers:
            max_inliers = cur_inliers
            best_H = H

        # If exceeds threshold exit loop
        if max_inliers > (len(match_pairs) * RANSAC_THRESH):
            break

    return best_H


def merge_images(cur_image, next_image, H):
    next_image_matrix = transpose_copy(next_image, "image")
    new_row, new_col = (cur_image.shape[1] + next_image.shape[1], cur_image.shape[0])
    transformed_matrix = np.zeros((new_row, new_col, next_image_matrix.shape[2]))

    # Traverse image pixels to calculate new indices
    for i in range(next_image_matrix.shape[0]):
        for j in range(next_image_matrix.shape[1]):
            dot_product = np.dot(H, [i, j, 1])
            i_match = int(dot_product[0, 0] / dot_product[0, 2] + 0.5)
            j_match = int(dot_product[0, 1] / dot_product[0, 2] + 0.5)
            if 0 <= i_match < new_row and 0 <= j_match < new_col:
                transformed_matrix[i_match, j_match] = next_image_matrix[i, j]

    transformed_next_image = transpose_copy(transformed_matrix, "matrix")
    plt.imshow(transformed_next_image)
    plt.title(""), plt.xticks([]), plt.yticks([])
    plt.show()

    # Find non black pixels in current image and create empty mask
    non_black_mask = np.all(cur_image != [0, 0, 0], axis=-1)
    empty_mask = np.zeros(
        (transformed_next_image.shape[0], transformed_next_image.shape[1]), dtype=bool
    )
    empty_mask[0 : cur_image.shape[0], 0 : cur_image.shape[1]] = non_black_mask

    # Assign non black pixels of current image to transformed next image
    transformed_next_image[empty_mask, :] = cur_image[non_black_mask, :]
    transformed_next_image = crop_image(transformed_next_image)
    plt.imshow(cv2.cvtColor(transformed_next_image, cv2.COLOR_BGR2RGB))
    plt.show()
    return transformed_next_image


def find_homography(match_pairs):
    matrix_list = []
    for pair in match_pairs:
        p1 = np.matrix([pair.item(0), pair.item(1), 1])
        p2 = np.matrix([pair.item(2), pair.item(3), 1])
        matrix_list.append(
            [
                0,
                0,
                0,
                -p2.item(2) * p1.item(0),
                -p2.item(2) * p1.item(1),
                -p2.item(2) * p1.item(2),
                p2.item(1) * p1.item(0),
                p2.item(1) * p1.item(1),
                p2.item(1) * p1.item(2),
            ]
        )

        matrix_list.append(
            [
                -p2.item(2) * p1.item(0),
                -p2.item(2) * p1.item(1),
                -p2.item(2) * p1.item(2),
                0,
                0,
                0,
                p2.item(0) * p1.item(0),
                p2.item(0) * p1.item(1),
                p2.item(0) * p1.item(2),
            ]
        )

    matrixA = np.matrix(matrix_list)
    u, s, vh = np.linalg.svd(matrixA)
    H = np.reshape(vh[8], (3, 3))
    H = (1 / H.item(8)) * H
    return H


# Calculate the geometric distance of matched points
def get_geometric_distance(match_pairs, H):
    estimated = np.dot(
        H, np.transpose(np.matrix([match_pairs[0].item(0), match_pairs[0].item(1), 1]))
    )
    # Eliminate division by zero and return inefficient d
    if estimated.item(2) == 0:
        return INLIER_THRESH + 1
    err = (
        np.transpose(np.matrix([match_pairs[0].item(2), match_pairs[0].item(3), 1]))
        - (1 / estimated.item(2)) * estimated
    )
    return np.linalg.norm(err)


def stitch_images(cur_image, cur_image_gray, next_image, next_image_gray):
    extractor = cv2.SIFT_create()
    feature_points_plot = None
    # Feature extraction
    cur_feature_pts, cur_descs, feature_points_plot = feature_extraction(
        cur_image, cur_image_gray, feature_points_plot, extractor
    )

    next_feature_pts, next_descs, feature_points_plot = feature_extraction(
        next_image, next_image_gray, feature_points_plot, extractor
    )

    # Feature matching
    matches = feature_matching(
        cur_image, cur_feature_pts, cur_descs, next_image, next_feature_pts, next_descs
    )

    # Find Homography matrix
    if matches is not None and matches.size != 0 and len(matches[:, 0]) >= 4:
        match_pairs_list = []
        for match in matches[:, 0]:
            (x1, y1) = next_feature_pts[match.queryIdx].pt
            (x2, y2) = cur_feature_pts[match.trainIdx].pt
            match_pairs_list.append([x1, y1, x2, y2])

        match_pairs_matrix = np.matrix(match_pairs_list)

        # Run RANSAC algorithm
        H = RANSAC(match_pairs_matrix)
        print("Homography: ", H)
        return H

    else:
        print("Can not find enough key points.")
        return None


def crop_image(image):
    # Crop top if black
    if not np.sum(image[0]):
        return crop_image(image[1:])

    # Crop bottom if black
    elif not np.sum(image[-1]):
        return crop_image(image[:-2])

    # Crop left if black
    elif not np.sum(image[:, 0]):
        return crop_image(image[:, 1:])

    # Crop right if black
    elif not np.sum(image[:, -1]):
        return crop_image(image[:, :-2])

    return image


def transpose_copy(copying, type):
    if type == "matrix":
        return np.transpose(copying.copy(), (1, 0, 2)).astype("uint8")
    else:
        return np.transpose(copying.copy(), (1, 0, 2))


def main():
    raw_dataset = get_subset_names(RAW_DATASET_PATH)
    # print(raw_dataset)

    for subset_name, subset_images in raw_dataset.items():
        homographies = []

        for idx in range(len(subset_images) - 1):
            print(idx, subset_images[idx])

            # Current image is the panorama
            cur_image = cv2.imread(subset_images[idx])
            cur_image_gray = cv2.cvtColor(cur_image, cv2.COLOR_BGR2GRAY)
            cur_image_gray = cv2.GaussianBlur(cur_image_gray, (3, 3), 0)

            # Read next images
            next_image = cv2.imread(subset_images[idx + 1])
            next_image_gray = cv2.cvtColor(next_image, cv2.COLOR_BGR2GRAY)
            next_image_gray = cv2.GaussianBlur(next_image_gray, (3, 3), 0)

            # Feature extraction, feature matching, Homography finding
            homography_matrix = stitch_images(
                cur_image,
                cur_image_gray,
                next_image,
                next_image_gray,
            )

            if homography_matrix is not None:
                homographies.append(homography_matrix)

        H = None
        panorama = cv2.imread(subset_images[0])
        print("Homographies", len(homographies))
        # Merging by transformation
        for i in range(len(homographies) - 1):
            print(subset_images[i])

            cur_image = cv2.imread(subset_images[i])
            next_image = cv2.imread(subset_images[i + 1])

            if H is None:
                H = homographies[0]
            else:
                H = np.matmul(H, homographies[i])

            panorama = merge_images(cur_image, next_image, H)
            plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
            plt.xticks([]), plt.yticks([])
            plt.show()

        # Apply median blur to the final panorama
        panorama = cv2.medianBlur(panorama, 3)

        # Display and save the panorama
        plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
        panorama_title = f"{subset_name} Panorama"
        plt.title(panorama_title), plt.xticks([]), plt.yticks([])

        # Specify the complete file path with folder, separator, and file extension
        output_path = os.path.join(OUTPUTS_PATH, "panorama_images")
        create_directory(output_path)
        output_file_path = os.path.join(output_path, f"{panorama_title}.png")

        # Save the panorama with the correct file path
        plt.savefig(output_file_path)
        plt.close()


if __name__ == "__main__":
    main()
