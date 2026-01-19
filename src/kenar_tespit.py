"""
kenar_tespit.py
This file demonstrates edge detection techniques used in image processing.
Sobel, Prewitt, and Canny methods are applied to a grayscale image to
detect edges and compare their results.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Image Loading

img_path= "/content/Goruntu2.jpg"
img= cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Sobel Edge Detection
# Compute gradients in x and y directions
sobel_x = cv2.Sobel(img,cv2.CV_64F,1,0, ksize=3)
sobel_y = cv2.Sobel(img, cv2.CV_64F,0,1, ksize=3)

# Gradient magnitude
sobel= cv2.magnitude(sobel_x, sobel_y)

# Prewitt Edge Detection
# Prewitt kernels
prewitt_kernel_x= np.array([
    [-1,  0,  1],
    [-1,  0,  1],
    [-1,  0,  1]
])

prewitt_kernel_y= np.array([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1]
])

# Apply kernels
prewitt_x= cv2.filter2D(img,-1,prewitt_kernel_x)
prewitt_y= cv2.filter2D(img,-1,prewitt_kernel_y)

# Gradient magnitude
prewitt= cv2.magnitude(
    prewitt_x.astype(float),
    prewitt_y.astype(float)
)

# Canny Edge Detection
canny= cv2.Canny(img,100,200)

# Visualization of Results
plt.figure(figsize=(10,5))
plt.imshow(img,cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.figure(figsize=(10,5))
plt.imshow(sobel,cmap='gray')
plt.title("Sobel Edge Detection")
plt.axis("off")

plt.figure(figsize=(10,5))
plt.imshow(prewitt,cmap='gray')
plt.title("Prewitt Edge Detection")
plt.axis("off")

plt.figure(figsize=(10,5))
plt.imshow(canny,cmap='gray')
plt.title("Canny Edge Detection")
plt.axis("off")

plt.show()
