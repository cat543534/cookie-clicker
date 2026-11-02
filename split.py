import os
import shutil
from pathlib import Path
import random

# Пункт 9: создание папок images и labels
base_dir = Path("/home/user/Рабочий стол/pechenka/itog")
base_dir.mkdir(exist_ok=True)
(base_dir / "images").mkdir(exist_ok=True)
(base_dir / "labels").mkdir(exist_ok=True)

# Перемещение файлов (пункт 9)
all_files = list(base_dir.iterdir())
for f in all_files:
    if f.is_file():
        if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
            shutil.move(str(f), str(base_dir / "images" / f.name))
        elif f.suffix.lower() == ".txt":
            shutil.move(str(f), str(base_dir / "labels" / f.name))

# Пункт 10: разделение на train (80%) и val (20%)
img_dir = base_dir / "images"
lbl_dir = base_dir / "labels"

img_files = [f for f in img_dir.iterdir() if f.is_file()]
random.shuffle(img_files)

split_idx = int(len(img_files) * 0.8)
train_imgs = img_files[:split_idx]
val_imgs = img_files[split_idx:]

(img_dir / "train").mkdir(exist_ok=True)
(img_dir / "val").mkdir(exist_ok=True)
(lbl_dir / "train").mkdir(exist_ok=True)
(lbl_dir / "val").mkdir(exist_ok=True)

for img in train_imgs:
    shutil.move(str(img), str(img_dir / "train" / img.name))
    lbl_path = lbl_dir / img.with_suffix(".txt").name
    if lbl_path.exists():
        shutil.move(str(lbl_path), str(lbl_dir / "train" / lbl_path.name))

for img in val_imgs:
    shutil.move(str(img), str(img_dir / "val" / img.name))
    lbl_path = lbl_dir / img.with_suffix(".txt").name
    if lbl_path.exists():
        shutil.move(str(lbl_path), str(lbl_dir / "val" / lbl_path.name))

# Пункт 11: проверка соответствия путей (картинка и label в одинаковых подпапках)
# (автоматически обеспечено логикой выше; блок ниже — валидация)
for subdir in ["train", "val"]:
    img_sub = img_dir / subdir
    lbl_sub = lbl_dir / subdir
    for img_file in img_sub.iterdir():
        if img_file.is_file():
            lbl_file = lbl_sub / img_file.with_suffix(".txt").name
            if not lbl_file.exists():
                print(f"WARNING: Отсутствует label для {img_file}")
