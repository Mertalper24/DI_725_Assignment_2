import os
import torch
import wandb
from tqdm import tqdm
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from dataset_class import train_dataloader, val_dataloader, processor

# Import the model configuration from your existing train.py
from transformers import DetrForObjectDetection

id2label = {
    0: "Human",  # Changed from "person" to match dataset
    1: "Car",
    2: "Truck",
    3: "Van",
    4: "Motorbike",
    5: "Bicycle",
    6: "Bus",
    7: "Trailer"
}
label2id = {v: k for k, v in id2label.items()}

# Initialize model
model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-50",
    num_labels=len(id2label),
    ignore_mismatched_sizes=True,
    id2label=id2label,
    label2id=label2id
)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training parameters
num_epochs = 10
learning_rate = 5e-6  # Reduced from 1e-5
weight_decay = 1e-4
warmup_steps = 100
max_grad_norm = 0.1  # Add gradient clipping

# Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=warmup_steps, 
    num_training_steps=total_steps
)

# Create output directory for checkpoints
os.makedirs("checkpoints", exist_ok=True)

# Training loop
best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
    for batch in progress_bar:
        # Move batch to device
        pixel_values = batch['pixel_values'].to(device)
        labels = [{k: v.to(device) if k != 'image_id' else v for k, v in label.items()} for label in batch['labels']]
        
        # Forward pass
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        scheduler.step()
        
        # Update metrics
        train_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})
        
        # Log to wandb
        wandb.log({"train_loss": loss.item()})
    
    avg_train_loss = train_loss / len(train_dataloader)
    
    # Validation
    model.eval()
    val_loss = 0
    val_progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
    
    with torch.no_grad():
        for batch in val_progress_bar:
            pixel_values = batch['pixel_values'].to(device)
            labels = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                      for k, v in label.items()} for label in batch['labels']]
            
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            
            val_loss += loss.item()
            val_progress_bar.set_postfix({"loss": loss.item()})
    
    avg_val_loss = val_loss / len(val_dataloader)
    
    # Log metrics to wandb
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "learning_rate": scheduler.get_last_lr()[0]
    })
    
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    # Save checkpoint if validation loss improved
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        checkpoint_path = f"checkpoints/detr_epoch_{epoch+1}_val_loss_{avg_val_loss:.4f}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': avg_val_loss,
        }, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

# Save the final model
final_model_path = "checkpoints/detr_final_model.pth"
torch.save(model.state_dict(), final_model_path)
print(f"Saved final model: {final_model_path}")

# Close wandb
wandb.finish() 