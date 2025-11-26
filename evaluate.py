"""
Evaluation Script - Compute mAP@0.5
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.diffusiondet import DiffusionDet
from utils.voc_dataset import VOCDataset, get_transform, collate_fn


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / (union + 1e-6)


def compute_ap(recalls, precisions):
    """Compute Average Precision"""
    # Add sentinel values
    mrec = np.concatenate(([0.], recalls, [1.]))
    mpre = np.concatenate(([0.], precisions, [0.]))
    
    # Compute precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    
    # Calculate area under PR curve
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    
    return ap


def evaluate_detections(all_predictions, all_targets, num_classes=20, iou_threshold=0.5):
    """
    Evaluate detections and compute mAP
    
    all_predictions: list of dicts with 'boxes', 'labels', 'scores'
    all_targets: list of dicts with 'boxes', 'labels'
    """
    aps = []
    
    for class_id in range(num_classes):
        # Collect all predictions and ground truths for this class
        class_dets = []
        class_gts = []
        
        for pred, target in zip(all_predictions, all_targets):
            # Predictions for this class
            pred_mask = pred['labels'] == class_id
            if pred_mask.any():
                class_dets.append({
                    'boxes': pred['boxes'][pred_mask].cpu().numpy(),
                    'scores': pred['scores'][pred_mask].cpu().numpy()
                })
            else:
                class_dets.append({'boxes': np.array([]), 'scores': np.array([])})
            
            # Ground truths for this class
            gt_mask = target['labels'] == class_id
            class_gts.append({
                'boxes': target['boxes'][gt_mask].cpu().numpy(),
                'num_boxes': gt_mask.sum().item()
            })
        
        # Compute AP for this class
        ap = compute_class_ap(class_dets, class_gts, iou_threshold)
        aps.append(ap)
    
    # Return mAP and per-class APs
    return np.mean(aps), aps


def compute_class_ap(detections, ground_truths, iou_threshold):
    """Compute AP for a single class"""
    # Collect all detections
    all_scores = []
    all_boxes = []
    all_image_ids = []
    
    for img_id, det in enumerate(detections):
        if len(det['scores']) > 0:
            all_scores.extend(det['scores'])
            all_boxes.extend(det['boxes'])
            all_image_ids.extend([img_id] * len(det['scores']))
    
    if len(all_scores) == 0:
        return 0.0
    
    # Sort by score
    sorted_indices = np.argsort(all_scores)[::-1]
    all_boxes = [all_boxes[i] for i in sorted_indices]
    all_image_ids = [all_image_ids[i] for i in sorted_indices]
    
    # Count total ground truth objects
    num_gt_objects = sum(gt['num_boxes'] for gt in ground_truths)
    
    if num_gt_objects == 0:
        return 0.0
    
    # Match predictions to ground truths
    tp = np.zeros(len(all_boxes))
    fp = np.zeros(len(all_boxes))
    matched_gts = [set() for _ in ground_truths]
    
    for pred_idx, (pred_box, img_id) in enumerate(zip(all_boxes, all_image_ids)):
        gt_boxes = ground_truths[img_id]['boxes']
        
        if len(gt_boxes) == 0:
            fp[pred_idx] = 1
            continue
        
        # Find best matching GT
        max_iou = 0
        max_gt_idx = -1
        
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gts[img_id]:
                continue
            
            iou = calculate_iou(pred_box, gt_box)
            if iou > max_iou:
                max_iou = iou
                max_gt_idx = gt_idx
        
        # Check if match is valid
        if max_iou >= iou_threshold and max_gt_idx not in matched_gts[img_id]:
            tp[pred_idx] = 1
            matched_gts[img_id].add(max_gt_idx)
        else:
            fp[pred_idx] = 1
    
    # Compute precision and recall
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    recalls = tp_cumsum / num_gt_objects
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    
    # Compute AP
    ap = compute_ap(recalls, precisions)
    
    return ap


def evaluate_model(model, data_loader, device, num_classes=20):
    """Evaluate model on dataset"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    print("Running inference...")
    with torch.no_grad():
        for images, targets in tqdm(data_loader):
            images = torch.stack([img.to(device) for img in images])
            
            # Get predictions
            predictions = model(images)
            
            all_predictions.extend(predictions)
            all_targets.extend(targets)
    
    print("Computing mAP...")
    mAP, aps = evaluate_detections(all_predictions, all_targets, num_classes)
    
    return mAP, aps


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth')
    parser.add_argument('--data-path', type=str, default='data/VOCdevkit/VOC2007')
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading test dataset...")
    test_dataset = VOCDataset(
        root=args.data_path,
        image_set='test',
        transforms=get_transform(train=False)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    model = DiffusionDet(num_classes=20)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Evaluate
    mAP, aps = evaluate_model(model, test_loader, device)
    
    # Print results
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"\nmAP@0.5: {mAP*100:.2f}%")
    
    print("\nPer-class AP:")
    voc_classes = VOCDataset.VOC_CLASSES
    for i, (cls, ap) in enumerate(zip(voc_classes, aps)):
        print(f"  {cls:<15} : {ap*100:>5.2f}%")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    main()
