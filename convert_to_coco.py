import json
import os
from tqdm import tqdm
from collections import defaultdict

INPUT_JSON = "/home/ai/Downloads/PhD/auair2019/annotations.json"
IMAGES_DIR = "/home/ai/Downloads/PhD/auair2019/images"  # Folder containing the image files
OUTPUT_JSON = "/home/ai/Downloads/PhD/auair2019/auair_coco.json"

# Category mapping (index: name)
# Note: DETR expects class indices to start from 0
category_names = [
    "Human", "Car", "Truck", "Van", "Motorbike",
    "Bicycle", "Bus", "Trailer"
]
categories = [
    {"id": i, "name": name} for i, name in enumerate(category_names)
]

def convert():
    with open(INPUT_JSON) as f:
        data = json.load(f)

    images = []
    annotations = []
    ann_id = 1
    added_imgs = set()

    for i, ann in enumerate(tqdm(data["annotations"])):
        try:
            image_name = ann["image_name"]
            # Extract numeric ID from filename
            image_id = int(image_name.split("_")[-1].split(".")[0])
            
            # Fix the typo in the original dataset (note the colon in "image_width:")
            width = int(ann.get("image_width:", 1920))
            height = int(ann.get("image_height", 1080))

            if image_id not in added_imgs:
                images.append({
                    "id": image_id,
                    "file_name": image_name,
                    "width": width,
                    "height": height
                })
                added_imgs.add(image_id)

            for box in ann["bbox"]:
                # Original format has top, left, height, width
                top = int(box["top"])
                left = int(box["left"])
                height = int(box["height"])
                width = int(box["width"])
                
                # COCO format uses [x, y, width, height] where x,y is top-left corner
                x = left
                y = top
                w = width
                h = height
                
                # Important: Adjust class index to be 0-based
                # The original dataset uses 1-based indexing (Human=0, Car=1, etc.)
                # But we want 0-based (Human=0, Car=1, etc.)
                class_id = int(box["class"])
                
                # Skip invalid boxes
                if w <= 0 or h <= 0:
                    continue
                
                annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": class_id,  # Using the original class ID
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0
                })
                ann_id += 1
        except KeyError as e:
            print(f"Error processing annotation {i}: {e}")
            print(f"Annotation content: {ann}")
            continue

    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(coco_dict, f, indent=4)
    print(f"COCO JSON saved to {OUTPUT_JSON}")
    print(f"Converted {len(images)} images and {len(annotations)} annotations")
    print(f"Category mapping: {categories}")

if __name__ == "__main__":
    convert()
