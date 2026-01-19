"""
histogram_tabanli_islemler.py
This file demonstrates histogram-based image processing techniques.
It includes grayscale histogram visualization, brightness and contrast
adjustment, gamma correction, and histogram equalization.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Image Loading

img_path= "/content/Goruntu1.jpg"
img= cv2.imread(img_path)
gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Original Image and Histogram

plt.figure(figsize=(10,5))
plt.imshow(gray, cmap='gray')
plt.title("Original Grayscale Image")
plt.axis('off')

plt.figure(figsize=(4,2))
plt.hist(gray.ravel(), bins=256,range=[0,256])
plt.title("Original Histogram")

# Brightness and Contrast Adjustment


bright_contrast= cv2.convertScaleAbs(gray,alpha=1.5,beta=50)

plt.figure(figsize=(10,5))
plt.imshow(bright_contrast, cmap='gray')
plt.title("Brightness and Contrast Adjustment")
plt.axis('off')

plt.figure(figsize=(4,2))
plt.hist(bright_contrast.ravel(),bins=256,range=[0, 256])
plt.title("Histogram (Brightness + Contrast)")

# Gamma Correction

for gamma in [0.5,1.0,2.0]:
    gamma_correction= np.array(
        255 *(gray/255) ** gamma, dtype='uint8'
    )

    plt.figure(figsize=(10, 5))
    plt.imshow(gamma_correction, cmap='gray')
    plt.title(f"Gamma Correction (γ = {gamma})")
    plt.axis('off')
    plt.show()

# Histogram of the last gamma corrected image
plt.figure(figsize=(4, 2))
plt.hist(gamma_correction.ravel(), bins=256, range=[0, 256])
plt.title("Histogram (Gamma Correction)")
plt.tight_layout()
plt.show()

# Histogram Equalization

equalized = cv2.equalizeHist(gray)

plt.figure(figsize=(10,5))
plt.imshow(equalized, cmap='gray')
plt.title("Histogram Equalization")
plt.axis('off')

plt.figure(figsize=(4,2))
plt.hist(equalized.ravel(),bins=256,range=[0,256])
plt.title("Histogram (Equalization)")
plt.show()
