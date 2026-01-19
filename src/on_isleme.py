import cv2
import numpy as np
import glob
import os

image_paths=sorted(glob.glob("/content/sample_data/*.jpg"))
images=[]

print("Görsel Özellikleri")
for path in image_paths:
    img = cv2.imread(path)

    h, w, c=img.shape
    print(f"{os.path.basename(path)}->boyut:{w}x{h},kanal:{c},dtype:{img.dtype},dinamik aralık:[{img.min()}-{img.max()}]")

    images.append(img)

# Ortak boyut icin ilk gorsel referans
target_h,target_w=images[0].shape[:2]

print("\nYeniden Örnekleme ve Ortak Boyuta Getirme")
new_img=[]

for i,img in enumerate(images):
    new=cv2.resize(img, (target_w,target_h),interpolation=cv2.INTER_AREA)
    new_img.append(new)

    output_path=f"/content/new_{i+1}.jpg"
    cv2.imwrite(output_path,new)
    print(f"{output_path} kaydedildi ({target_w}x{target_h})")
