"""
Simplified DiffusionDet Implementation for VOC 2007
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
import math

class DiffusionDet(nn.Module):
    """
    Simplified DiffusionDet for object detection
    """
    def __init__(self, num_classes=20, num_proposals=300, num_timesteps=1000, 
                 d_model=256, nhead=8, num_layers=6):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_proposals = num_proposals
        self.num_timesteps = num_timesteps
        self.d_model = d_model
        
        # Backbone
        resnet = resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Reduce channels
        self.input_proj = nn.Conv2d(2048, d_model, kernel_size=1)
        
        # Positional encoding for features
        self.pos_encoder = PositionalEncoding2D(d_model)
        
        # Box and time embeddings
        self.box_embed = MLP(4, d_model, d_model, 3)
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, 
            dim_feedforward=2048, dropout=0.1, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Prediction heads
        self.class_head = nn.Linear(d_model, num_classes + 1)
        self.bbox_head = MLP(d_model, d_model, 4, 3)
        
        # Diffusion schedule
        self.setup_diffusion_schedule(num_timesteps)
    
    def setup_diffusion_schedule(self, timesteps):
        """Setup cosine beta schedule"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0.0001, 0.9999)
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
    
    def extract(self, a, t, shape):
        """Extract values at timestep t from 1D buffer `a`."""
        # t should be int64 and on the same device as `a`
        t = t.long()
        if t.device != a.device:
            t = t.to(a.device)

        b = t.shape[0]

        # a is 1D: (T,), t is (B,) -> out is (B,)
        out = a.gather(0, t)  # both on same device now

        # Reshape to broadcast over x_start
        return out.view(b, *([1] * (len(shape) - 1)))
    
    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion process"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alpha_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alpha_t * x_start + sqrt_one_minus_alpha_t * noise
    
    def forward(self, images, targets=None):
        """
        Forward pass
        images: (B, 3, H, W)
        targets: list of dicts with 'boxes' and 'labels'
        """
        # Extract features
        features = self.backbone(images)
        features = self.input_proj(features)
        B, C, H, W = features.shape
        
        if self.training and targets is not None:
            # Prepare targets
            gt_boxes, gt_labels, valid_mask = self.prepare_targets(targets, B)
            
            # Sample timestep
            t = torch.randint(0, self.num_timesteps, (B,), device=images.device).long()
            
            # Add noise to boxes
            noise = torch.randn_like(gt_boxes)
            noisy_boxes = self.q_sample(gt_boxes, t, noise)
            
            # Predict
            pred_boxes, pred_logits = self.denoise(noisy_boxes, t, features)
            
            # Compute losses
            losses = self.compute_losses(pred_boxes, pred_logits, gt_boxes, gt_labels, valid_mask)
            return losses
        else:
            # Inference
            return self.inference(features, images.device)
    
    def denoise(self, boxes, t, features):
        """Single denoising step"""
        B, C, H, W = features.shape
        
        # Box embeddings
        box_embed = self.box_embed(boxes)  # (B, N, d_model)
        
        # Time embeddings
        time_embed = self.time_embed(t)  # (B, d_model)
        time_embed = time_embed.unsqueeze(1).expand(-1, self.num_proposals, -1)
        
        # Query = box + time
        query = box_embed + time_embed
        
        # Memory from features
        features_flat = features.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        pos_embed = self.pos_encoder(features).flatten(2).permute(0, 2, 1)
        memory = features_flat + pos_embed
        
        # Decode
        output = self.decoder(query, memory)  # (B, N, d_model)
        
        # Predictions
        pred_boxes = self.bbox_head(output).sigmoid()
        pred_logits = self.class_head(output)
        
        return pred_boxes, pred_logits
    
    @torch.no_grad()
    def inference(self, features, device):
        """Generate predictions via reverse diffusion"""
        B = features.shape[0]
        
        # Start from noise
        boxes = torch.randn(B, self.num_proposals, 4, device=device)
        
        # Reverse diffusion
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            pred_boxes, pred_logits = self.denoise(boxes, t, features)
            
            # Update boxes (simplified DDPM sampling)
            if i > 0:
                noise = torch.randn_like(boxes)
                boxes = pred_boxes + 0.1 * noise
            else:
                boxes = pred_boxes
        
        # Final predictions
        scores = pred_logits.softmax(-1)
        labels = scores.argmax(-1)
        confidence = scores.max(-1)[0]
        
        # Format output
        detections = []
        for b in range(B):
            # Filter background and low confidence
            keep = (labels[b] < self.num_classes) & (confidence[b] > 0.05)
            detections.append({
                'boxes': boxes[b][keep],
                'labels': labels[b][keep],
                'scores': confidence[b][keep]
            })
        
        return detections
    
    def prepare_targets(self, targets, batch_size):
        """Prepare and pad targets"""
        gt_boxes_list = []
        gt_labels_list = []
        valid_masks = []
        
        for t in targets:
            boxes = t['boxes']
            labels = t['labels']
            
            num_boxes = len(boxes)
            if num_boxes > self.num_proposals:
                # Randomly sample if too many
                indices = torch.randperm(num_boxes)[:self.num_proposals]
                boxes = boxes[indices]
                labels = labels[indices]
                num_boxes = self.num_proposals
            
            # Pad
            padded_boxes = torch.zeros(self.num_proposals, 4, device=boxes.device)
            padded_labels = torch.full((self.num_proposals,), self.num_classes, 
                                      dtype=torch.long, device=labels.device)
            valid_mask = torch.zeros(self.num_proposals, device=boxes.device)
            
            padded_boxes[:num_boxes] = boxes
            padded_labels[:num_boxes] = labels
            valid_mask[:num_boxes] = 1
            
            gt_boxes_list.append(padded_boxes)
            gt_labels_list.append(padded_labels)
            valid_masks.append(valid_mask)
        
        return (torch.stack(gt_boxes_list), 
                torch.stack(gt_labels_list),
                torch.stack(valid_masks))
    
    def compute_losses(self, pred_boxes, pred_logits, gt_boxes, gt_labels, valid_mask):
        """Compute training losses"""
        # Box loss (L1)
        loss_bbox = F.l1_loss(pred_boxes, gt_boxes, reduction='none')
        loss_bbox = (loss_bbox.sum(-1) * valid_mask).sum() / valid_mask.sum()
        
        # Classification loss
        loss_cls = F.cross_entropy(
            pred_logits.reshape(-1, self.num_classes + 1),
            gt_labels.reshape(-1),
            reduction='mean'
        )
        
        # GIoU loss (simplified)
        loss_giou = self.giou_loss(pred_boxes, gt_boxes, valid_mask)
        
        return {
            'loss_bbox': loss_bbox,
            'loss_cls': loss_cls,
            'loss_giou': loss_giou,
            'loss_total': loss_bbox + loss_cls + 2.0 * loss_giou
        }
    
    def giou_loss(self, pred, target, valid_mask):
        """Simplified GIoU loss"""
        # This is a simplified version - you can improve it
        x1 = torch.max(pred[..., 0], target[..., 0])
        y1 = torch.max(pred[..., 1], target[..., 1])
        x2 = torch.min(pred[..., 2], target[..., 2])
        y2 = torch.min(pred[..., 3], target[..., 3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        pred_area = (pred[..., 2] - pred[..., 0]) * (pred[..., 3] - pred[..., 1])
        target_area = (target[..., 2] - target[..., 0]) * (target[..., 3] - target[..., 1])
        
        union = pred_area + target_area - intersection
        iou = intersection / (union + 1e-6)
        
        loss = (1 - iou) * valid_mask
        return loss.sum() / valid_mask.sum()


# Helper modules
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            elif i == num_layers - 1:
                layers.append(nn.Linear(hidden_dim, output_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            
            if i < num_layers - 1:
                layers.append(nn.ReLU())
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Create position encodings
        y_pos = torch.arange(H, device=x.device, dtype=x.dtype).unsqueeze(1).repeat(1, W)
        x_pos = torch.arange(W, device=x.device, dtype=x.dtype).unsqueeze(0).repeat(H, 1)
        
        # Normalize
        y_pos = y_pos / H
        x_pos = x_pos / W
        
        # Create sinusoidal encodings
        div_term = torch.exp(torch.arange(0, C//2, 2, device=x.device, dtype=x.dtype) * 
                            -(math.log(10000.0) / (C//2)))
        
        pos_encoding = torch.zeros(H, W, C, device=x.device, dtype=x.dtype)
        
        pos_encoding[:, :, 0::4] = torch.sin(x_pos[:, :, None] * div_term)
        pos_encoding[:, :, 1::4] = torch.cos(x_pos[:, :, None] * div_term)
        pos_encoding[:, :, 2::4] = torch.sin(y_pos[:, :, None] * div_term)
        pos_encoding[:, :, 3::4] = torch.cos(y_pos[:, :, None] * div_term)
        
        pos_encoding = pos_encoding.permute(2, 0, 1).unsqueeze(0)
        
        return pos_encoding
    