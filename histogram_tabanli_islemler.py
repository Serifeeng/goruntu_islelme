import cv2
import numpy as np
import matplotlib.pyplot as plt

img_path= "/content/Goruntu1.jpg"
img= cv2.imread(img_path)
gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Orijinal histogram
plt.figure(figsize=(10,5))

plt.imshow(gray, cmap='gray')
plt.title("Orijinal Gri Görsel")
plt.axis('off')

plt.figure(figsize=(4,2))
plt.hist(gray.ravel(), bins=256, range=[0,256])
plt.title("Orijinal Histogram")

#Parlaklik ve kontrast duzenleme
bright_contrast= cv2.convertScaleAbs(gray, alpha=1.5, beta=50)

plt.figure(figsize=(10,5))
plt.imshow(bright_contrast, cmap='gray')
plt.title("Parlaklık+Kontrast")
plt.axis('off')

plt.figure(figsize=(4,2))
plt.hist(bright_contrast.ravel(), bins=256, range=[0,256])
plt.title("Histogram (Parlaklık+Kontrast)")

#Gamma duzeltmesi
for gamma in [0.5, 1.0, 2.0]:
    gamma_correction = np.array(255*(gray/255)**gamma, dtype='uint8')
    plt.figure(figsize=(10,5))
    plt.imshow(gamma_correction, cmap='gray')
    plt.title(f"Gamma= {gamma}")
    plt.axis('off')
    plt.show()

plt.figure(figsize=(4,2))
plt.hist(gamma_correction.ravel(), bins=256, range=[0,256])
plt.title("Histogram (Gamma)")

plt.tight_layout()
plt.show()

#Histogram esitleme
equalized= cv2.equalizeHist(gray)

plt.figure(figsize=(10,5))
plt.imshow(equalized, cmap='gray')
plt.title("Histogram Eşitleme")
plt.axis('off')

plt.figure(figsize=(4,2))
plt.hist(equalized.ravel(), bins=256, range=[0,256])
plt.title("Histogram (Eşitleme)")
plt.show()