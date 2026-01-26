import os
import shutil

SOURCE_ROOT = r"D:\Defect_Scanner\Dataset" #D:\Defect_Scanner     D:\Defect_Scanner\Dataset
DEST_ROOT = r"D:\Defect_Scanner\merged_dataset"            #D:\Defect_Scanner\merged_dataset

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg"]

os.makedirs(os.path.join(DEST_ROOT, "images"), exist_ok=True)
os.makedirs(os.path.join(DEST_ROOT, "labels"), exist_ok=True)

counter = 0

for folder in sorted(os.listdir(SOURCE_ROOT)):
    folder_path = os.path.join(SOURCE_ROOT, folder)

    if not os.path.isdir(folder_path):
        continue

    for file in sorted(os.listdir(folder_path)):
        name, ext = os.path.splitext(file)

        if ext.lower() not in IMAGE_EXTENSIONS:
            continue

        img_src = os.path.join(folder_path, file)
        lbl_src = os.path.join(folder_path, name + ".txt")

        new_base = f"frame_{counter:06d}"

        # copy image (ALWAYS)
        shutil.copy(
            img_src,
            os.path.join(DEST_ROOT, "images", new_base + ext)
        )

        # copy label ONLY if it exists
        if os.path.exists(lbl_src):
            shutil.copy(
                lbl_src,
                os.path.join(DEST_ROOT, "labels", new_base + ".txt")
            )

        counter += 1

print(f"✅ DONE: {counter} images merged (labels copied only if present)")
