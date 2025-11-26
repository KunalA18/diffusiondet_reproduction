"""
Train Faster R-CNN Baseline
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
import os
import sys
import torchvision.transforms as T

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.voc_dataset import VOCDataset, get_transform, collate_fn


def get_faster_rcnn_model(num_classes):
    """Get Faster R-CNN model with ResNet-50"""
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    
    # Replace box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    
    return model


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(data_loader, desc=f'Epoch {epoch}')
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
        pbar.set_postfix({'loss': f"{losses.item():.4f}"})
    
    return total_loss / len(data_loader)

def get_transform_frcnn(train=True):
    # Faster R-CNN expects images as tensors in [0,1], variable size.
    # Don't resize here; don't normalize here.
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)


def main():
    config = {
        'num_epochs': 2,
        'batch_size': 2,
        'learning_rate': 0.005,
        'save_dir': 'checkpoints_frcnn',
        'data_path': 'data/VOCdevkit/VOC2007'
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Load dataset
    print("Loading datasets...")
    train_dataset = VOCDataset(
        root=config['data_path'],
        image_set='trainval',
        transforms=get_transform_frcnn(train=True)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    # Create model
    print("Creating Faster R-CNN model...")
    model = get_faster_rcnn_model(num_classes=20)
    model = model.to(device)
    
    # Optimizer and scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=config['learning_rate'], 
                         momentum=0.9, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # Training loop
    print("\nStarting training...")
    print("=" * 60)
    
    best_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        avg_loss = train_one_epoch(model, optimizer, train_loader, device, epoch + 1)
        
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        print(f"  Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0 or avg_loss < best_loss:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_path = os.path.join(config['save_dir'], 'best_model.pth')
                print(f"  Saving best model (loss: {best_loss:.4f})")
            else:
                save_path = os.path.join(config['save_dir'], f'checkpoint_epoch_{epoch+1}.pth')
            
            torch.save(checkpoint, save_path)
        
        scheduler.step()
        print("-" * 60)
    
    print(f"\nTraining complete! Best loss: {best_loss:.4f}")


if __name__ == '__main__':
    main()