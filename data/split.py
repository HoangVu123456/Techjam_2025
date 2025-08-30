import os
import random
import shutil

def split_labelme_dataset(
    input_dir,
    output_dir,
    train_ratio=0.8,
    seed=42
):
    """
    Split a dataset so that each base file (e.g., bug.5001) and all its extensions (.jpg, .html, .json)
    are always copied together into train or test.
    """
    random.seed(seed)

    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Group files by base name (without extension)
    files = os.listdir(input_dir)
    base_names = set(os.path.splitext(f)[0] for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    base_names = sorted(base_names)
    random.shuffle(base_names)

    split_idx = int(len(base_names) * train_ratio)
    train_bases = base_names[:split_idx]
    test_bases = base_names[split_idx:]

    def copy_group(bases, dest_dir):
        for base in bases:
            for ext in ['.jpg', '.jpeg', '.png', '.html','.xml', '.json']:
                file = base + ext
                src = os.path.join(input_dir, file)
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(dest_dir, file))

    copy_group(train_bases, train_dir)
    copy_group(test_bases, test_dir)

    print(f"✅ Split complete: {len(train_bases)} train, {len(test_bases)} test")
    print(f"Train folder: {train_dir}")
    print(f"Test folder: {test_dir}")

if __name__ == "__main__":
    input_folder = r"F:\fasterrcnn_resnet50_fpn_v2_new_dataset\data\ppe\5k_6k_labelme\default"
    output_folder = r"F:\fasterrcnn_resnet50_fpn_v2_new_dataset\data\ppe\labelme_split"
    split_labelme_dataset(input_folder, output_folder, train_ratio=0.8)