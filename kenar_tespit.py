import cv2
import numpy as np
import matplotlib.pyplot as plt

img_path= "/content/Goruntu2.jpg"
img= cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Sobel
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobel_x, sobel_y)

# Prewitt filtreleri 
prewitt_kernel_x = np.array([[ -1, 0, 1],
                             [ -1, 0, 1],
                             [ -1, 0, 1]])

prewitt_kernel_y = np.array([[ -1, -1, -1],
                             [  0,  0,  0],
                             [  1,  1,  1]])

prewitt_x = cv2.filter2D(img, -1, prewitt_kernel_x)
prewitt_y = cv2.filter2D(img, -1, prewitt_kernel_y)
prewitt = cv2.magnitude(prewitt_x.astype(float), prewitt_y.astype(float))

# Canny
canny = cv2.Canny(img, 100, 200)

# Sonuçları çizdir
plt.figure(figsize=(10,5))
plt.imshow(img, cmap='gray')
plt.title("Orijinal")
plt.axis("off")

plt.figure(figsize=(10,5))
plt.imshow(sobel, cmap='gray')
plt.title("Sobel")
plt.axis("off")

plt.figure(figsize=(10,5))
plt.imshow(prewitt, cmap='gray')
plt.title("Prewitt")
plt.axis("off")

plt.figure(figsize=(10,5))
plt.imshow(canny, cmap='gray')
plt.title("Canny")
plt.axis("off")

plt.show()