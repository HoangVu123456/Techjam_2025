import os
import shutil

def split_images_labels(source_dir, output_dir):
    """
    Split a messy folder into separate 'images' and 'labels' folders.

    Args:
        source_dir (str): Path to messy folder containing images + txt labels.
        output_dir (str): Path to save cleaned dataset structure.
    """
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    for file in os.listdir(source_dir):
        src_path = os.path.join(source_dir, file)

        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            shutil.copy(src_path, os.path.join(images_dir, file))
        elif file.lower().endswith(".txt"):
            shutil.copy(src_path, os.path.join(labels_dir, file))

    print(f"✅ Done! Images in '{images_dir}', Labels in '{labels_dir}'")

if __name__ == "__main__":
    split_images_labels(
        source_dir=os.path.normpath("D:\Repo\TikTok_Hackathon\Techjam_2025\data\yolo_data\yes"),
        output_dir="D:\Repo\TikTok_Hackathon\Techjam_2025\data\yolo_data"
    )