import os
import random
import shutil

def split_labelme_dataset_exact(
    input_dir,
    output_dir,
    train_count=1600,
    test_count=350,
    seed=42
):
    """
    Split a LabelMe dataset into train, test, and leftover with exact counts.
    Each split contains all file types for a given base name.
    """
    random.seed(seed)

    # Output directories
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    leftover_dir = os.path.join(output_dir, "leftover")

    for d in [train_dir, test_dir, leftover_dir]:
        os.makedirs(d, exist_ok=True)

    # Get all image base names
    files = os.listdir(input_dir)
    base_names = [os.path.splitext(f)[0] for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if train_count + test_count > len(base_names):
        raise ValueError("train_count + test_count exceeds total number of images!")

    random.shuffle(base_names)
    train_bases = base_names[:train_count]
    test_bases = base_names[train_count:train_count+test_count]
    leftover_bases = base_names[train_count+test_count:]

    def copy_group(bases, dest_dir):
        for base in bases:
            for ext in ['.jpg', '.jpeg', '.png', '.json', '.xml', '.html']:
                src = os.path.join(input_dir, base + ext)
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(dest_dir, base + ext))

    copy_group(train_bases, train_dir)
    copy_group(test_bases, test_dir)
    copy_group(leftover_bases, leftover_dir)

    print(f"✅ Split complete: {len(train_bases)} train, {len(test_bases)} test, {len(leftover_bases)} leftover")
    print(f"Train folder: {train_dir}")
    print(f"Test folder: {test_dir}")
    print(f"Leftover folder: {leftover_dir}")


if __name__ == "__main__":
    input_folder = r"D:\Repo\TikTok_Hackathon\Techjam_2025\data\raw_data\combination"
    output_folder = r"D:\Repo\TikTok_Hackathon\Techjam_2025\test_data\combination"
    split_labelme_dataset_exact(input_folder, output_folder,
                                train_count=1600, test_count=350)
