"""
Script to create the Visual Wake Words dataset from COCO 2017.

For each image in COCO, we assign a binary label as follows:
  - Label 1 if the image contains at least one bounding box (for the “person” category)
    with area >= (threshold * image_area) [default threshold=0.002 i.e. 0.2%].
  - Label 0 otherwise.

The final dataset is saved in an ImageNet folder format:
    output_dir/
        train/
            0/  # images with no “person” bounding boxes above threshold
            1/  # images with at least one “person” above threshold
        val/
            0/
            1/

Usage:
    python create_visual_wake_words.py --download --output_dir ./visual_wake_words

Note: Downloading and extracting COCO requires significant disk space.
"""

import os
import argparse
import requests
import zipfile
import json
import shutil
from tqdm import tqdm


def download_file(url, output_path):
    """Download a file from a URL with progress display."""
    if os.path.exists(output_path):
        print(f"{output_path} already exists; skipping download.")
        return

    print(f"Downloading {url} to {output_path}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    with open(output_path, 'wb') as file:
        for data in tqdm(response.iter_content(block_size),
                         total=total_size // block_size, unit='KB', desc=os.path.basename(output_path)):
            file.write(data)
    print("Download complete.")


def extract_zip(zip_path, extract_to):
    """Extract a ZIP file to a given directory."""
    print(f"Extracting {zip_path} to {extract_to}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete.")


def create_imagenet_structure(base_dir, splits, labels):
    """Create directory structure in the ImageNet format."""
    for split in splits:
        for label in labels:
            dir_path = os.path.join(base_dir, split, str(label))
            os.makedirs(dir_path, exist_ok=True)
            print(f"Created directory: {dir_path}")


def process_coco_split(split, images_dir, annotations_file, output_dir, threshold):
    """
    Process a COCO split (train/val).

    For each image, load its metadata from the annotation file, check for person bounding boxes
    that meet the area threshold, and copy the image to the appropriate folder.
    """
    print(f"\nProcessing the {split} split...")
    # Load the annotation JSON file.
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)

    # Create a dictionary mapping image_id -> image info.
    images_info = {img['id']: img for img in coco_data['images']}

    # Build a mapping from image_id to a list of person annotations.
    # In COCO, the category_id for "person" is 1.
    person_annotations = {}
    for ann in coco_data['annotations']:
        if ann['category_id'] == 1:
            img_id = ann['image_id']
            person_annotations.setdefault(img_id, []).append(ann)

    # Process each image.
    for img_id, img_info in tqdm(images_info.items(), desc=f"Processing {split} images"):
        img_width = img_info['width']
        img_height = img_info['height']
        image_area = img_width * img_height

        label = 0  # Default label.
        anns = person_annotations.get(img_id, [])
        for ann in anns:
            # Bounding box format: [x, y, width, height]
            bbox = ann['bbox']
            box_area = bbox[2] * bbox[3]
            if box_area >= threshold * image_area:
                label = 1
                break

        # Construct source and destination paths.
        src_img_path = os.path.join(images_dir, img_info['file_name'])
        dst_img_path = os.path.join(output_dir, split, str(label), img_info['file_name'])

        # Copy the image file.
        try:
            shutil.copy(src_img_path, dst_img_path)
        except Exception as e:
            print(f"Error copying {src_img_path} to {dst_img_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Create Visual Wake Words dataset from COCO 2017.")
    parser.add_argument('--output_dir', type=str, default='visual_wake_words',
                        help="Output directory for the dataset in ImageNet format.")
    parser.add_argument('--threshold', type=float, default=0.002,
                        help="Fraction of image area for a bounding box to be considered (default: 0.002 for 0.2%).")
    parser.add_argument('--download', action='store_true',
                        help="Download COCO dataset zips if not present.")
    args = parser.parse_args()

    # Directories for COCO data and output dataset.
    coco_dir = 'coco'
    os.makedirs(coco_dir, exist_ok=True)
    output_dir = args.output_dir

    # URLs for COCO 2017 files.
    train_images_url = "http://images.cocodataset.org/zips/train2017.zip"
    val_images_url = "http://images.cocodataset.org/zips/val2017.zip"
    annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

    train_zip_path = os.path.join(coco_dir, "train2017.zip")
    val_zip_path = os.path.join(coco_dir, "val2017.zip")
    annotations_zip_path = os.path.join(coco_dir, "annotations_trainval2017.zip")

    # Download the zip files if requested.
    if args.download:
        download_file(train_images_url, train_zip_path)
        download_file(val_images_url, val_zip_path)
        download_file(annotations_url, annotations_zip_path)

    # Extract the datasets (if not already extracted).
    train_extract_dir = os.path.join(coco_dir, "train2017")
    val_extract_dir = os.path.join(coco_dir, "val2017")
    annotations_extract_dir = os.path.join(coco_dir, "annotations")

    if not os.path.isdir(train_extract_dir):
        extract_zip(train_zip_path, coco_dir)
    if not os.path.isdir(val_extract_dir):
        extract_zip(val_zip_path, coco_dir)
    if not os.path.isdir(annotations_extract_dir):
        extract_zip(annotations_zip_path, coco_dir)

    # Create the output directory structure (ImageNet-style).
    splits = ['train', 'val']
    labels = [0, 1]
    create_imagenet_structure(output_dir, splits, labels)

    # Define annotation file paths.
    train_annotations_file = os.path.join(annotations_extract_dir, "instances_train2017.json")
    val_annotations_file = os.path.join(annotations_extract_dir, "instances_val2017.json")

    # Process both the train and validation splits.
    process_coco_split('train', train_extract_dir, train_annotations_file, output_dir, args.threshold)
    process_coco_split('val', val_extract_dir, val_annotations_file, output_dir, args.threshold)

    print("\nVisual Wake Words dataset creation complete!")
    print(f"Dataset saved in ImageNet format under: {os.path.abspath(output_dir)}")


if __name__ == '__main__':
    main()
