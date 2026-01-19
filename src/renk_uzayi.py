"""
renk_uzayi_parlaklik_ayarlari.py
This script demonstrates brightness adjustment in different color spaces
(HSV and YCbCr) and compares the results with the original RGB image.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Görüntü yolu
img_path="/content/Goruntu4.jpg"

# Görüntüyü BGR formatında oku
img= cv2.imread(img_path)

# OpenCV BGR kullandığı için RGB'ye çevir
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# RGB görüntüyü HSV ve YCbCr renk uzaylarına dönüştür
hsv=cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
ycbcr=cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)

#HSV PARLAKLIK AYARI

# HSV görüntünün kopyası
hsv_adj=hsv.copy()

# V (Value) kanalı parlaklığı temsil eder
V=hsv_adj[:, :, 2].astype(float)

# Parlaklık artırımı (taşmayı önlemek için clip)
V=np.clip(V + 100, 0, 255)

# Güncellenmiş V kanalını geri ata
hsv_adj[:, :, 2] = V.astype(np.uint8)

# HSV'den tekrar RGB'ye dönüş
hsv_to_rgb=cv2.cvtColor(hsv_adj, cv2.COLOR_HSV2RGB)

#YCbCr PARLAKLIK AYARI 

# YCbCr görüntünün kopyası
ycbcr_adj=ycbcr.copy()

# Y kanalı parlaklık bilgisini taşır
Y=ycbcr_adj[:, :, 0].astype(float)

# Parlaklık artırımı
Y=np.clip(Y + 20, 0, 255)

# Güncellenmiş Y kanalını geri ata
ycbcr_adj[:, :, 0]=Y.astype(np.uint8)

# YCbCr'den tekrar RGB'ye dönüş
ycbcr_to_rgb=cv2.cvtColor(ycbcr_adj, cv2.COLOR_YCrCb2RGB)

# GÖRSELLEŞTİRME

plt.figure(figsize=(10,5))
plt.imshow(img_rgb)
plt.title("Orijinal RGB")
plt.axis("off")

plt.figure(figsize=(10,5))
plt.imshow(hsv_to_rgb)
plt.title("HSV'de V Parlaklık Artışı")
plt.axis("off")

plt.figure(figsize=(10,5))
plt.imshow(ycbcr_to_rgb)
plt.title("YCbCr'de Y Parlaklık Artışı")
plt.axis("off")

plt.figure(figsize=(10,5))
plt.imshow(hsv)
plt.title("HSV Renk Uzayı")
plt.axis("off")

plt.figure(figsize=(10,5))
plt.imshow(ycbcr)
plt.title("YCbCr Renk Uzayı")
plt.axis("off")

plt.show()
