import cv2
import numpy as np
import matplotlib.pyplot as plt

img_path= "/content/Goruntu2.jpg"
img= cv2.imread(img_path)
img_gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Laplacian Tabanli Keskinlestirme
laplacian= cv2.Laplacian(img_gray, cv2.CV_64F)
laplacian= cv2.convertScaleAbs(laplacian)

sharp_laplacian= cv2.addWeighted(img_gray, 1, laplacian, -1, 0)

# Unsharp Masking
blur= cv2.GaussianBlur(img_gray, (5,5), 1)
mask= img_gray - blur
k= 1
unsharp= cv2.addWeighted(img_gray, 1, mask, k, 0)

# High-Boost Filtreleme
blur_hb= cv2.GaussianBlur(img_gray, (5,5), 1)
mask_hb= img_gray - blur_hb
# Farklı A değerleri
A_values= [1.0, 1.5, 3.0]
highboost_results= []

for A in A_values:
    hb= cv2.addWeighted(img_gray, A, mask_hb, 1, 0)
    highboost_results.append(hb)

plt.figure(figsize=(12,6))
plt.imshow(img_gray, cmap='gray')
plt.title("Original")
plt.axis("off")

plt.figure(figsize=(12,6))
plt.imshow(sharp_laplacian, cmap='gray')
plt.title("Laplacian Sharpened")
plt.axis("off")

plt.figure(figsize=(12,6))
plt.imshow(unsharp, cmap='gray')
plt.title(f"Unsharp Masking (k={k})")
plt.axis("off")

#High-boost sonuçları
for i, (A, hb_img) in enumerate(zip(A_values, highboost_results)):
    plt.figure(figsize=(12,6))
    plt.imshow(hb_img, cmap='gray')
    plt.title(f"High-Boost (A={A})")
    plt.axis("off")

plt.tight_layout()
plt.show()