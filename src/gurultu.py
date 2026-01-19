import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# Metrikler

def mse(img1, img2):
    return np.mean((img1.astype("float32") - img2.astype("float32")) ** 2)

def psnr(img1, img2):
    mse_val = mse(img1, img2)
    if mse_val == 0:
        return 999
    return 20 * np.log10(255.0 / np.sqrt(mse_val))

# Gürültü Fonksiyonları

# Gaussian Noise
def add_gaussian_noise(image, mean=0, var=40):
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, image.shape).astype("float32")
    noisy = image.astype("float32") + gauss
    return np.clip(noisy, 0, 255).astype("uint8")

# Salt & Pepper Noise
def add_salt_pepper_noise(image, amount=0.2):
    noisy = image.copy()
    h, w = image.shape
    num = int(amount * h * w)

    # Salt
    y = np.random.randint(0, h, num)
    x = np.random.randint(0, w, num)
    noisy[y, x] = 255

    # Pepper
    y = np.random.randint(0, h, num)
    x = np.random.randint(0, w, num)
    noisy[y, x] = 0

    return noisy
paths = sorted([
    "/content/Goruntu1.jpg",
    "/content/Goruntu3.jpg"
])


for path in paths:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    filename = os.path.basename(path)

    # Gürültü ekleme
    gauss_noise = add_gaussian_noise(img)
    sp_noise = add_salt_pepper_noise(img)

    # Filtreler
    mean_gauss = cv2.blur(gauss_noise, (5, 5))
    mean_sp = cv2.blur(sp_noise, (5, 5))

    gauss_gauss = cv2.GaussianBlur(gauss_noise, (5, 5), 1)
    gauss_sp = cv2.GaussianBlur(sp_noise, (5, 5), 1)

    median_gauss = cv2.medianBlur(gauss_noise, 5)
    median_sp = cv2.medianBlur(sp_noise, 5)

    # MSE ve PSNR
    results = {
        "Mean (Gaussian Noise)": (mse(img, mean_gauss), psnr(img, mean_gauss)),
        "Mean (Salt&Pepper)": (mse(img, mean_sp), psnr(img, mean_sp)),

        "Gaussian (Gaussian Noise)": (mse(img, gauss_gauss), psnr(img, gauss_gauss)),
        "Gaussian (Salt&Pepper)": (mse(img, gauss_sp), psnr(img, gauss_sp)),

        "Median (Gaussian Noise)": (mse(img, median_gauss), psnr(img, median_gauss)),
        "Median (Salt&Pepper)": (mse(img, median_sp), psnr(img, median_sp)),
    }

    print(f"\n{filename} için MSE ve PSNR")
    for k, (m, p) in results.items():
        print(f"{k} --> MSE: {m:.2f}, PSNR: {p:.2f}")

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 3, 1); plt.imshow(img, cmap='gray'); plt.title("Orijinal"); plt.axis("off")
    plt.subplot(3, 3, 2); plt.imshow(gauss_noise, cmap='gray'); plt.title("Gaussian Gürültü"); plt.axis("off")
    plt.subplot(3, 3, 3); plt.imshow(mean_gauss, cmap='gray'); plt.title("Mean Filter"); plt.axis("off")

    plt.subplot(3, 3, 4); plt.imshow(gauss_gauss, cmap='gray'); plt.title("Gaussian Filter"); plt.axis("off")
    plt.subplot(3, 3, 5); plt.imshow(median_gauss, cmap='gray'); plt.title("Median Filter"); plt.axis("off")

    plt.subplot(3, 3, 6); plt.imshow(sp_noise, cmap='gray'); plt.title("Salt & Pepper Gürültü"); plt.axis("off")
    plt.subplot(3, 3, 7); plt.imshow(mean_sp, cmap='gray'); plt.title("Mean Filter"); plt.axis("off")
    plt.subplot(3, 3, 8); plt.imshow(gauss_sp, cmap='gray'); plt.title("Gaussian Filter"); plt.axis("off")
    plt.subplot(3, 3, 9); plt.imshow(median_sp, cmap='gray'); plt.title("Median Filter"); plt.axis("off")

    plt.suptitle(f"Gürültü Azaltma Sonuçları – {filename}", fontsize=14)
    plt.tight_layout()

    plt.show()
