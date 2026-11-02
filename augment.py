import os
import cv2
import albumentations as A
from pathlib import Path

INPUT_DIR = "screenshots"
OUTPUT_DIR = "screenshots_augmented"
os.makedirs(OUTPUT_DIR, exist_ok=True)

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.05,
        scale_limit=0.1,
        rotate_limit=15,
        p=0.5
    ),
    A.Perspective(scale=(0.02, 0.08), p=0.3),
    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.5
    ),
    A.HueSaturationValue(
        hue_shift_limit=10,
        sat_shift_limit=20,
        val_shift_limit=20,
        p=0.4
    ),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.MotionBlur(blur_limit=5, p=0.2),
    A.OpticalDistortion(distort_limit=0.1, shift_limit=0.05, p=0.3),
    A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.3),
    A.RandomResizedCrop(size=(640, 640), scale=(0.8, 1.0), p=0.4)
])

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
files = [f for f in os.listdir(INPUT_DIR) if Path(f).suffix.lower() in image_extensions]

print(f"Найдено изображений: {len(files)}")

NUM_AUGS = 5
for idx, filename in enumerate(files):
    img_path = os.path.join(INPUT_DIR, filename)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for i in range(NUM_AUGS):
        transformed = transform(image=image)
        aug_image = transformed["image"]

        out_name = f"{Path(filename).stem}_aug{i}_{idx}.jpg"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        cv2.imwrite(out_path, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))

    if (idx + 1) % 10 == 0:
        print(f"Обработано: {idx + 1}/{len(files)}")

print(f"Готово! Аугментированные изображения в: {OUTPUT_DIR}")
