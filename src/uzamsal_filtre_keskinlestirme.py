"""
uzamsal_filtre_keskinlestirme.py
This script demonstrates image sharpening techniques using:
- Laplacian-based sharpening
- Unsharp Masking
- High-Boost filtering
All methods are applied on a grayscale image and visualized for comparison.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Görüntü yolu
img_path= "/content/Goruntu2.jpg"

# Görüntüyü oku ve gri seviyeye çevir
img=cv2.imread(img_path)
img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#LAPLACIAN TABANLI KESKİNLEŞTİRME

# Laplacian operatörü (ikinci türev tabanlı kenar vurgulama)
laplacian=cv2.Laplacian(img_gray, cv2.CV_64F)

# Mutlak değere çevirip görüntü formatına dönüştür
laplacian=cv2.convertScaleAbs(laplacian)

# Orijinal görüntüden Laplacian çıkarılarak keskinleştirme yapılır
sharp_laplacian=cv2.addWeighted(img_gray, 1, laplacian, -1, 0)

#UNSHARP MASKING

# Görüntüyü Gauss filtresi ile bulanıklaştır
blur=cv2.GaussianBlur(img_gray,(5, 5), 1)

# Mask: Orijinal - bulanık görüntü
mask=img_gray-blur

# Keskinlik katsayısı
k = 1

# Unsharp masking uygulanması
unsharp=cv2.addWeighted(img_gray, 1, mask, k, 0)

#HIGH-BOOST FİLTRELEME

# High-boost için bulanık görüntü
blur_hb=cv2.GaussianBlur(img_gray, (5, 5), 1)

# High-boost maskesi
mask_hb=img_gray-blur_hb

# Farklı A katsayıları (keskinlik seviyesi)
A_values=[1.0, 1.5, 3.0]
highboost_results=[]

for A in A_values:
    hb = cv2.addWeighted(img_gray, A, mask_hb, 1, 0)
    highboost_results.append(hb)

#GÖRSELLEŞTİRME

plt.figure(figsize=(12,6))
plt.imshow(img_gray, cmap='gray')
plt.title("Original Grayscale Image")
plt.axis("off")

plt.figure(figsize=(12,6))
plt.imshow(sharp_laplacian, cmap='gray')
plt.title("Laplacian-Based Sharpening")
plt.axis("off")

plt.figure(figsize=(12,6))
plt.imshow(unsharp, cmap='gray')
plt.title(f"Unsharp Masking (k={k})")
plt.axis("off")

# High-boost sonuçlarını çizdir
for A, hb_img in zip(A_values, highboost_results):
    plt.figure(figsize=(12,6))
    plt.imshow(hb_img, cmap='gray')
    plt.title(f"High-Boost Filtering (A={A})")
    plt.axis("off")

plt.tight_layout()
plt.show()
