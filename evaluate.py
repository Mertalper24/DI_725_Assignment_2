import os
import torch
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from transformers import DetrForObjectDetection, DetrImageProcessor
from dataset_class import AUAIRDataset, collate_fn

# Configuration
ANNOTATIONS_JSON = "/home/ai/Downloads/PhD/auair2019/annotations.json"
IMAGES_DIR = "/home/ai/Downloads/PhD/auair2019/images"
CHECKPOINT_PATH = "checkpoints/detr_epoch_10_val_loss_0.7528.pth"  # Use your best checkpoint
OUTPUT_DIR = "evaluation_results"
VISUALIZATION_DIR = os.path.join(OUTPUT_DIR, "visualizations")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

# Class mapping
id2label = {
    0: "Human",
    1: "Car",
    2: "Truck",
    3: "Van",
    4: "Motorbike",
    5: "Bicycle",
    6: "Bus",
    7: "Trailer"
}
label2id = {v: k for k, v in id2label.items()}

# Colors for visualization (RGB format for PIL)
colors = [
    (255, 0, 0),    # Red
    (0, 0, 255),    # Blue
    (0, 255, 0),    # Green
    (255, 165, 0),  # Orange
    (128, 0, 128),  # Purple
    (0, 255, 255),  # Cyan
    (255, 0, 255),  # Magenta
    (255, 255, 0)   # Yellow
]

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model and processor
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-50",
    num_labels=len(id2label),
    ignore_mismatched_sizes=True,
    id2label=id2label,
    label2id=label2id
)

# Load checkpoint
print(f"Loading checkpoint from {CHECKPOINT_PATH}")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Create dataset and dataloader
dataset = AUAIRDataset(ANNOTATIONS_JSON, IMAGES_DIR, processor)

# Create a larger subset of the dataset
subset_size = 500  # Adjust this number as needed
indices = list(range(subset_size))
large_dataset = Subset(dataset, indices)

# Use the larger dataset in the DataLoader
dataloader = DataLoader(large_dataset, batch_size=2, collate_fn=collate_fn)

print(f"Dataset size: {len(dataset)}")

# Function to visualize predictions using PIL instead of matplotlib
def visualize_predictions_pil(image, predictions, ground_truth, threshold=0.7):
    """
    Visualize predictions and ground truth on the image using PIL
    """
    # Create a copy of the image to draw on
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    width, height = image.size
    
    # Try to load a font, use default if not available
    try:
        # Increase font size from 15 to 24 for better visibility
        font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        try:
            # Try to load a system font with larger size
            font = ImageFont.load_default().font_variant(size=24)
        except:
            # If all else fails, use default font
            font = ImageFont.load_default()
    
    # Draw ground truth boxes with dashed lines (approximated in PIL)
    for label, box in zip(ground_truth["class_labels"], ground_truth["boxes"]):
        # Make sure label is an integer
        label_idx = label.cpu().item() if isinstance(label, torch.Tensor) else label
        
        # Denormalize box coordinates
        x0, y0, x1, y1 = box.tolist()
        x0 = int(x0 * width)
        y0 = int(y0 * height)
        x1 = int(x1 * width)
        y1 = int(y1 * height)
        
        # Draw dashed rectangle (approximated by drawing multiple small lines)
        color = colors[label_idx % len(colors)]
        dash_length = 10
        
        # Top line
        for i in range(x0, x1, dash_length*2):
            end = min(i + dash_length, x1)
            draw.line([(i, y0), (end, y0)], fill=color, width=2)
        
        # Right line
        for i in range(y0, y1, dash_length*2):
            end = min(i + dash_length, y1)
            draw.line([(x1, i), (x1, end)], fill=color, width=2)
        
        # Bottom line
        for i in range(x0, x1, dash_length*2):
            end = min(i + dash_length, x1)
            draw.line([(i, y1), (end, y1)], fill=color, width=2)
        
        # Left line
        for i in range(y0, y1, dash_length*2):
            end = min(i + dash_length, y1)
            draw.line([(x0, i), (x0, end)], fill=color, width=2)
        
        # Add ground truth label (smaller and at bottom of box)
        class_name = id2label[label_idx]
        text = f"{class_name}"
        text_size = draw.textbbox((0, 0), text, font=font)[2:4]
        
        # Draw text background
        draw.rectangle([(x0, y1), (x0 + text_size[0] + 4, y1 + text_size[1] + 4)], fill=color)
        draw.text((x0 + 2, y1 + 2), text, fill=(255, 255, 255), font=font)
    
    # Draw prediction boxes with solid lines
    for score, label, box in zip(predictions["scores"], predictions["labels"], predictions["boxes"]):
        if score >= threshold:
            # Make sure label is an integer
            label_idx = label.cpu().item() if isinstance(label, torch.Tensor) else label
            
            # Denormalize box coordinates
            x0, y0, x1, y1 = box.tolist()
            x0 = int(x0 * width)
            y0 = int(y0 * height)
            x1 = int(x1 * width)
            y1 = int(y1 * height)
            
            # Draw rectangle
            color = colors[label_idx % len(colors)]
            draw.rectangle([(x0, y0), (x1, y1)], outline=color, width=2)
            
            # Add label
            class_name = id2label[label_idx]
            text = f"{class_name}: {score:.2f}"
            text_size = draw.textbbox((0, 0), text, font=font)[2:4]
            
            # Draw text background
            draw.rectangle([(x0, y0 - text_size[1] - 4), (x0 + text_size[0] + 4, y0)], fill=color)
            draw.text((x0 + 2, y0 - text_size[1] - 2), text, fill=(255, 255, 255), font=font)
    
    return draw_image

# Function to calculate IoU
def box_iou(boxes1, boxes2):
    """
    Calculate IoU between two sets of boxes
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    
    union = area1[:, None] + area2 - inter
    
    iou = inter / union
    return iou

# Function to calculate mAP
def calculate_map(all_predictions, all_targets, iou_threshold=0.3):
    """
    Calculate mAP for all classes
    """
    # Initialize variables
    class_metrics = {i: {"TP": 0, "FP": 0, "FN": 0, "precision": 0, "recall": 0, "AP": 0} 
                    for i in range(len(id2label))}
    
    # Process each image
    for preds, targets in zip(all_predictions, all_targets):
        pred_boxes = preds["boxes"]
        pred_scores = preds["scores"]
        pred_labels = preds["labels"]
        
        gt_boxes = targets["boxes"]
        gt_labels = targets["class_labels"]
        
        # Skip if no predictions or ground truth
        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            continue
        
        # Normalize predicted boxes
        width, height = gt_boxes.shape[-2], gt_boxes.shape[-1]
        pred_boxes[:, [0, 2]] /= width
        pred_boxes[:, [1, 3]] /= height
        
        # Calculate IoU for all prediction-ground truth pairs
        ious = box_iou(pred_boxes, gt_boxes)
        
        # For each ground truth box, find the best matching prediction
        gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool)
        
        # Sort predictions by confidence
        sorted_indices = torch.argsort(pred_scores, descending=True)
        
        for idx in sorted_indices:
            # Make sure label is an integer and on CPU
            pred_label = pred_labels[idx].cpu().item() if isinstance(pred_labels[idx], torch.Tensor) else pred_labels[idx]
            
            # Find ground truth boxes with the same class
            same_class_mask = (gt_labels == pred_label)
            
            if not same_class_mask.any():
                # No ground truth with this class, it's a false positive
                class_metrics[pred_label]["FP"] += 1
                continue
            
            # Get IoUs with ground truth boxes of the same class
            valid_ious = ious[idx][same_class_mask]
            valid_gt_indices = torch.where(same_class_mask)[0]
            
            # Find the best matching ground truth box
            if valid_ious.shape[0] > 0:
                best_iou, best_idx = valid_ious.max(dim=0)
                best_gt_idx = valid_gt_indices[best_idx]
                
                if best_iou >= iou_threshold and not gt_matched[best_gt_idx]:
                    # True positive
                    class_metrics[pred_label]["TP"] += 1
                    gt_matched[best_gt_idx] = True
                else:
                    # False positive (either low IoU or already matched)
                    class_metrics[pred_label]["FP"] += 1
            else:
                # No valid ground truth, false positive
                class_metrics[pred_label]["FP"] += 1
        
        # Count false negatives (unmatched ground truth boxes)
        for i, matched in enumerate(gt_matched):
            if not matched:
                # Make sure label is an integer and on CPU
                gt_label = gt_labels[i].cpu().item() if isinstance(gt_labels[i], torch.Tensor) else gt_labels[i]
                class_metrics[gt_label]["FN"] += 1
    
    # Calculate precision, recall, and AP for each class
    overall_ap = 0
    valid_classes = 0
    
    for class_id, metrics in class_metrics.items():
        tp = metrics["TP"]
        fp = metrics["FP"]
        fn = metrics["FN"]
        
        # Skip classes with no ground truth examples
        if tp + fn == 0:
            continue
        
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        
        # Simple AP calculation (precision at max recall)
        ap = precision if recall > 0 else 0
        
        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["AP"] = ap
        
        overall_ap += ap
        valid_classes += 1
    
    # Calculate mAP
    mAP = overall_ap / valid_classes if valid_classes > 0 else 0
    
    return class_metrics, mAP

# Evaluation function
def evaluate_model(model, dataloader, processor, num_visualizations=10):
    model.eval()
    all_preds = []
    all_targets = []
    
    # For visualization
    vis_count = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels']  # Keep labels on CPU for now
            
            # Forward pass
            outputs = model(pixel_values=pixel_values)
            
            # Process predictions
            target_sizes = torch.tensor([[img.shape[-2], img.shape[-1]] for img in pixel_values])
            results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.1)
            
            for i, (result, label) in enumerate(zip(results, labels)):
                # Get original image for visualization
                if vis_count < num_visualizations:
                    try:
                        # Get image ID safely
                        if 'image_id' in label:
                            idx = label['image_id'].item()
                        else:
                            # Use a fallback index
                            idx = vis_count % len(dataset.annotations)
                        
                        # Get image path
                        ann = dataset.annotations[idx]
                        image_name = ann["image_name"]
                        img_path = os.path.join(IMAGES_DIR, image_name)
                        
                        # Check if file exists
                        if not os.path.exists(img_path):
                            print(f"Warning: Image file not found: {img_path}")
                            continue
                        
                        # Load image
                        image = Image.open(img_path).convert("RGB")
                        width, height = image.size
                        print(f"Image {image_name} dimensions: {width}x{height}")
                        
                        # Move tensors to CPU for visualization
                        cpu_result = {
                            "boxes": result["boxes"].cpu(),
                            "labels": result["labels"].cpu(),
                            "scores": result["scores"].cpu()
                        }
                        
                        cpu_label = {
                            "boxes": label["boxes"].cpu(),
                            "class_labels": label["class_labels"].cpu()
                        }
                        
                        # Visualize predictions using PIL
                        try:
                            vis_image = visualize_predictions_pil(image, cpu_result, cpu_label)
                            
                            # Save visualization
                            output_path = os.path.join(VISUALIZATION_DIR, f"pred_{vis_count}.png")
                            vis_image.save(output_path)
                            
                            print(f"Saved visualization to {output_path}")
                            vis_count += 1
                        except Exception as e:
                            print(f"Error visualizing prediction: {e}")
                    except Exception as e:
                        print(f"Error processing image for visualization: {e}")
                
                # Store predictions and targets
                pred_boxes = result["boxes"].cpu()
                pred_labels = result["labels"].cpu()
                pred_scores = result["scores"].cpu()
                
                # Get ground truth
                gt_boxes = label["boxes"].cpu()
                gt_labels = label["class_labels"].cpu()
                
                # Normalize predicted boxes
                pred_boxes[:, [0, 2]] /= width
                pred_boxes[:, [1, 3]] /= height
                
                # Calculate IoU
                ious = box_iou(pred_boxes, gt_boxes)
                
                # Store predictions and targets
                all_preds.append({
                    "boxes": pred_boxes,
                    "labels": pred_labels,
                    "scores": pred_scores
                })
                
                all_targets.append({
                    "boxes": gt_boxes,
                    "class_labels": gt_labels
                })
    
    # Calculate mAP
    class_metrics, mAP = calculate_map(all_preds, all_targets)
    
    return all_preds, all_targets, class_metrics, mAP

# Run evaluation
print("Evaluating model...")
all_preds, all_targets, class_metrics, mAP = evaluate_model(model, dataloader, processor)

# Print results
print("\nEvaluation Results:")
print(f"mAP@0.5: {mAP:.4f}")
print("\nPer-class metrics:")
print(f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'AP':<10}")
print("-" * 40)

for class_id, metrics in class_metrics.items():
    class_name = id2label[class_id]
    print(f"{class_name:<10} {metrics['precision']:.4f}      {metrics['recall']:.4f}      {metrics['AP']:.4f}")

# Save results to file
results = {
    "mAP": mAP,
    "class_metrics": {id2label[class_id]: metrics for class_id, metrics in class_metrics.items()}
}

with open(os.path.join(OUTPUT_DIR, "evaluation_results.json"), "w") as f:
    json.dump(results, f, indent=4)

print(f"\nResults saved to {os.path.join(OUTPUT_DIR, 'evaluation_results.json')}")
print(f"Visualizations saved to {VISUALIZATION_DIR}") 