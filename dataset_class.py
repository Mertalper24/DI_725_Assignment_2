import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import wandb
from transformers import DetrImageProcessor

# Initialize wandb
wandb.init(project="auair-object-detection", name="detr-finetune")

# Dataset class for original AUAIR format
class AUAIRDataset(Dataset):
    def __init__(self, annotations_json, img_dir, processor):
        self.img_dir = img_dir
        self.processor = processor
        
        # Load annotations
        with open(annotations_json, 'r') as f:
            data = json.load(f)
        
        self.annotations = data["annotations"]
        self.categories = data["categories"]
        
        # Create category name to id mapping
        self.cat2id = {name: i for i, name in enumerate(self.categories)}
        
        # Print dataset info for debugging
        print(f"Loaded {len(self.annotations)} annotations")
        print(f"Categories: {self.categories}")
        print(f"Category mapping: {self.cat2id}")
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        
        # Get image info
        image_name = ann["image_name"]
        img_path = os.path.join(self.img_dir, image_name)
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        width = int(ann.get("image_width:", 1920))  # Note the colon in the original data
        height = int(ann.get("image_height", 1080))
        
        # Prepare boxes and labels
        boxes = []
        class_labels = []
        
        for box in ann["bbox"]:
            # Original format has top, left, height, width
            top = int(box["top"])
            left = int(box["left"])
            h = int(box["height"])
            w = int(box["width"])
            
            # Convert to [x_min, y_min, x_max, y_max]
            x_min = max(0, left)
            y_min = max(0, top)
            x_max = min(width, left + w)
            y_max = min(height, top + h)
            
            # Skip invalid boxes
            if x_min >= x_max or y_min >= y_max:
                continue
                
            # Normalize coordinates to [0, 1]
            x_min = x_min / width
            y_min = y_min / height
            x_max = x_max / width
            y_max = y_max / height
            
            boxes.append([x_min, y_min, x_max, y_max])
            class_labels.append(int(box["class"]))  # Use class directly from annotation
        
        # Skip images without valid annotations
        if len(boxes) == 0:
            # Return a dummy sample
            boxes = torch.tensor([[0, 0, 1, 1]], dtype=torch.float32)
            class_labels = torch.tensor([0], dtype=torch.long)
        else:
            # Convert to tensors
            boxes = torch.tensor(boxes, dtype=torch.float32)
            class_labels = torch.tensor(class_labels, dtype=torch.long)
        
        # Create target dictionary
        target = {
            'boxes': boxes,
            'class_labels': class_labels,
            'image_id': torch.tensor([idx])  # Use index as image_id
        }
        
        # Apply processor
        encoding = self.processor(images=image, return_tensors="pt")
        encoding = {key: val.squeeze() for key, val in encoding.items()}
        encoding['labels'] = target
        
        # Debug first few samples
        if idx < 5:
            print(f"Sample {idx}:")
            print(f"  Image: {image_name}")
            print(f"  Boxes: {boxes}")
            print(f"  Labels: {class_labels}")
        
        return encoding

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([item['pixel_values'] for item in batch]),
        'labels': [item['labels'] for item in batch]
    }

# Setup data paths
ANNOTATIONS_JSON = "/home/ai/Downloads/PhD/auair2019/annotations.json"
IMAGES_DIR = "/home/ai/Downloads/PhD/auair2019/images"

# Load processor
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

# Create dataset and dataloader
dataset = AUAIRDataset(ANNOTATIONS_JSON, IMAGES_DIR, processor)

# Split into train and validation sets (80/20 split)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create data loaders
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=4, collate_fn=collate_fn)

print(f"Training dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
