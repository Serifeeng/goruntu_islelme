"""
goruntu_ozellikleri_ve_yeniden_ornekleme.py
This file analyzes basic image properties such as resolution, number of
channels, data type, and dynamic range. It also performs resampling by
resizing multiple images to a common resolution.
"""

import cv2
import numpy as np
import glob
import os

# Image Loading
image_paths= sorted(glob.glob("/content/sample_data/*.jpg"))
images= []

print("Image Properties")
for path in image_paths:
    img = cv2.imread(path)

    h, w, c= img.shape
    print(
        f"{os.path.basename(path)} -> "
        f"Resolution: {w}x{h}, "
        f"Channels: {c}, "
        f"Data type: {img.dtype}, "
        f"Dynamic range: [{img.min()}-{img.max()}]"
    )

    images.append(img)

# Resampling to Common Size
# Use the first image as reference for target resolution
target_h, target_w = images[0].shape[:2]

print("\nResampling Images to a Common Resolution")
resized_images= []

for i, img in enumerate(images):
    resized= cv2.resize(
        img,
        (target_w, target_h),
        interpolation=cv2.INTER_AREA
    )

    resized_images.append(resized)

    output_path= f"/content/new_{i+1}.jpg"
    cv2.imwrite(output_path, resized)
    print(f"{output_path} saved ({target_w}x{target_h})")
