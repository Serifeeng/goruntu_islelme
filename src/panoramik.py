"""
panoramic_image_stitching.py
This script creates a panoramic image by stitching multiple overlapping
images using OpenCV's Stitcher module.
"""

import cv2
import os
def load_images(image_folder):
    """
    Loads all PNG images from the given folder.
    Parameters:
        image_folder (str): Path to the folder containing input images.
    Returns:
        list: A list of loaded images.
    """
    images=[]

    for file_name in sorted(os.listdir(image_folder)):
        if file_name.lower().endswith(".png"):
            path= os.path.join(image_folder, file_name)
            img= cv2.imread(path)

            if img is None:
                print(f"Error: {file_name} could not be read.")
            else:
                images.append(img)

    return images


def stitch_images(images):
    """
    Stitches multiple images into a single panoramic image.
    Parameters:
        images (list): List of images to be stitched.
    Returns:
        ndarray: The resulting panoramic image.
    Raises:
        RuntimeError: If panorama creation fails.
    """
    stitcher= cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    status, panorama = stitcher.stitch(images)

    if status==cv2.Stitcher_OK:
        return panorama
    else:
        raise RuntimeError(f"Panorama creation failed. Error code: {status}")


if __name__ == "__main__":
    image_folder="images"
    output_path="panor_
