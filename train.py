"""
Main Training Script for DiffusionDet
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.diffusiondet import DiffusionDet
from utils.voc_dataset import VOCDataset, get_transform, collate_fn


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    total_bbox_loss = 0
    total_cls_loss = 0
    total_giou_loss = 0
    
    pbar = tqdm(data_loader, desc=f'Epoch {epoch}')
    for images, targets in pbar:
        # Move to device
        images = torch.stack([img.to(device) for img in images])
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward
        losses = model(images, targets)
        loss = losses['loss_total']
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Track losses
        total_loss += loss.item()
        total_bbox_loss += losses['loss_bbox'].item()
        total_cls_loss += losses['loss_cls'].item()
        total_giou_loss += losses['loss_giou'].item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'bbox': f"{losses['loss_bbox'].item():.4f}",
            'cls': f"{losses['loss_cls'].item():.4f}"
        })
    
    # Return average losses
    n = len(data_loader)
    return {
        'total': total_loss / n,
        'bbox': total_bbox_loss / n,
        'cls': total_cls_loss / n,
        'giou': total_giou_loss / n
    }


def main():
    # Configuration
    config = {
        'num_epochs': 2,
        'batch_size': 2,
        'learning_rate': 1e-4,
        'num_proposals': 300,
        'num_timesteps': 1000,
        'save_dir': 'checkpoints',
        'data_path': 'data/VOCdevkit/VOC2007'
    }
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = VOCDataset(
        root=config['data_path'],
        image_set='trainval',
        transforms=get_transform(train=True)
    )
    
    print(f"Training samples: {len(train_dataset)}")
    
    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Create model
    print("Creating model...")
    model = DiffusionDet(
        num_classes=20,
        num_proposals=config['num_proposals'],
        num_timesteps=config['num_timesteps']
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=1e-4
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs']
    )
    
    # Training loop
    print("\nStarting training...")
    print("=" * 60)
    
    history = {
        'total': [],
        'bbox': [],
        'cls': [],
        'giou': []
    }
    
    best_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        # Train
        losses = train_one_epoch(model, optimizer, train_loader, device, epoch + 1)
        
        # Update history
        for key in history:
            history[key].append(losses[key])
        
        # Print epoch summary
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        print(f"  Total Loss: {losses['total']:.4f}")
        print(f"  BBox Loss:  {losses['bbox']:.4f}")
        print(f"  Class Loss: {losses['cls']:.4f}")
        print(f"  GIoU Loss:  {losses['giou']:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0 or losses['total'] < best_loss:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': losses['total'],
                'config': config
            }
            
            if losses['total'] < best_loss:
                best_loss = losses['total']
                save_path = os.path.join(config['save_dir'], 'best_model.pth')
                print(f"  Saving best model (loss: {best_loss:.4f})")
            else:
                save_path = os.path.join(config['save_dir'], f'checkpoint_epoch_{epoch+1}.pth')
            
            torch.save(checkpoint, save_path)
        
        # Update learning rate
        scheduler.step()
        print("-" * 60)
    
    # Plot training curves
    print("\nPlotting training curves...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(history['total'])
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(history['bbox'], label='BBox')
    axes[0, 1].plot(history['giou'], label='GIoU')
    axes[0, 1].set_title('Box Losses')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(history['cls'])
    axes[1, 0].set_title('Classification Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True)
    
    # Combined view
    axes[1, 1].plot(history['total'], label='Total', linewidth=2)
    axes[1, 1].plot(history['bbox'], label='BBox', alpha=0.7)
    axes[1, 1].plot(history['cls'], label='Class', alpha=0.7)
    axes[1, 1].plot(history['giou'], label='GIoU', alpha=0.7)
    axes[1, 1].set_title('All Losses')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config['save_dir'], 'training_curves.png'), dpi=150)
    print(f"Saved training curves to {config['save_dir']}/training_curves.png")
    
    print("\nTraining complete!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Model saved to: {config['save_dir']}/best_model.pth")


if __name__ == '__main__':
    main()
