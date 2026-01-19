import cv2
import numpy as np
import matplotlib.pyplot as plt

img_path = "/content/Goruntu4.jpg"
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
ycbcr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)

hsv_adj = hsv.copy()
V = hsv_adj[:,:,2].astype(float)

V = np.clip(V + 100, 0, 255)
hsv_adj[:,:,2] = V.astype(np.uint8)

hsv_to_rgb = cv2.cvtColor(hsv_adj, cv2.COLOR_HSV2RGB)

ycbcr_adj = ycbcr.copy()
Y = ycbcr_adj[:,:,0].astype(float)

Y = np.clip(Y + 20, 0, 255)
ycbcr_adj[:,:,0] = Y.astype(np.uint8)

ycbcr_to_rgb = cv2.cvtColor(ycbcr_adj, cv2.COLOR_YCrCb2RGB)

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
