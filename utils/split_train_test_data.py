import os
import random
import shutil

def split_voc_dataset(
    images_dir,
    annotations_dir,
    output_dir,
    train_ratio=0.8,
    seed=42
):
    """
    Split VOC-style dataset into train and test.
    Each split will have its own 'images' and 'annotations' folders.
    """
    random.seed(seed)

    # Output directories
    train_images_dir = os.path.join(output_dir, "train", "images")
    train_ann_dir = os.path.join(output_dir, "train", "annotations")
    test_images_dir = os.path.join(output_dir, "test", "images")
    test_ann_dir = os.path.join(output_dir, "test", "annotations")

    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_ann_dir, exist_ok=True)
    os.makedirs(test_images_dir, exist_ok=True)
    os.makedirs(test_ann_dir, exist_ok=True)

    # Get all image base names
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    base_names = [os.path.splitext(f)[0] for f in image_files]

    random.shuffle(base_names)
    split_idx = int(len(base_names) * train_ratio)
    train_bases = base_names[:split_idx]
    test_bases = base_names[split_idx:]

    def copy_split(bases, src_images_dir, src_ann_dir, dst_images_dir, dst_ann_dir):
        for base in bases:
            # Copy image
            for ext in ['.jpg', '.jpeg', '.png']:
                src_img = os.path.join(src_images_dir, base + ext)
                if os.path.exists(src_img):
                    shutil.copy(src_img, os.path.join(dst_images_dir, base + ext))
            # Copy annotation
            src_xml = os.path.join(src_ann_dir, base + '.xml')
            if os.path.exists(src_xml):
                shutil.copy(src_xml, os.path.join(dst_ann_dir, base + '.xml'))

    copy_split(train_bases, images_dir, annotations_dir, train_images_dir, train_ann_dir)
    copy_split(test_bases, images_dir, annotations_dir, test_images_dir, test_ann_dir)

    print(f"✅ Split complete: {len(train_bases)} train, {len(test_bases)} test")
    print(f"Train images: {train_images_dir}, Train annotations: {train_ann_dir}")
    print(f"Test images: {test_images_dir}, Test annotations: {test_ann_dir}")


if __name__ == "__main__":
    images_folder = r"D:\Repo\TikTok_Hackathon\Techjam_2025\data\raw_data\images"
    annotations_folder = r"D:\Repo\TikTok_Hackathon\Techjam_2025\data\raw_data\annotations"
    output_folder = r"D:\Repo\TikTok_Hackathon\Techjam_2025\test_data"
    split_voc_dataset(images_folder, annotations_folder, output_folder, train_ratio=0.8)
