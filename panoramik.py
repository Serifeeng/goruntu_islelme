import cv2
import os

def load_images(image_folder):
    images = []
    for file_name in sorted(os.listdir(image_folder)):
        if file_name.lower().endswith(".png"):
            path = os.path.join(image_folder, file_name)
            img = cv2.imread(path)
            if img is None:
                print(f"Hata: {file_name} okunamadı.")
            else:
                images.append(img)
    return images


def stitch_images(images):
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    status, panorama = stitcher.stitch(images)

    if status == cv2.Stitcher_OK:
        return panorama
    else:
        raise RuntimeError(f"Panorama oluşturulamadi. Hata kodu: {status}")

if __name__ == "__main__":
    image_folder = "images"
    output_path = "panorama_result.png"

    images = load_images(image_folder)

    if len(images) < 2:
        raise ValueError("En az 2 görüntü gerekli.")

    panorama = stitch_images(images)

    cv2.imwrite(output_path, panorama)
    print("Panorama başarıyla oluşturuldu:", output_path)

    cv2.imshow("Panorama", panorama)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
