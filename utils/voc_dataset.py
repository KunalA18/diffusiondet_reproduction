"""
PASCAL VOC Dataset Loader
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import xml.etree.ElementTree as ET
import os
import torchvision.transforms as T

class VOCDataset(Dataset):
    """PASCAL VOC 2007 Dataset for Object Detection"""
    
    VOC_CLASSES = (
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    )
    
    def __init__(self, root, image_set='trainval', transforms=None):
        self.root = root
        self.image_set = image_set
        self.transforms = transforms
        
        self.class_to_idx = {cls: i for i, cls in enumerate(self.VOC_CLASSES)}
        
        # Load image IDs
        split_file = os.path.join(root, 'ImageSets', 'Main', f'{image_set}.txt')
        with open(split_file) as f:
            self.ids = [x.strip() for x in f.readlines()]
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        img_id = self.ids[idx]
        
        # Load image
        img_path = os.path.join(self.root, 'JPEGImages', f'{img_id}.jpg')
        img = Image.open(img_path).convert('RGB')
        
        # Load annotations
        anno_path = os.path.join(self.root, 'Annotations', f'{img_id}.xml')
        boxes, labels = self.parse_annotation(anno_path)
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Normalize boxes to [0, 1]
        w, h = img.size
        boxes[:, [0, 2]] /= w
        boxes[:, [1, 3]] /= h
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])
        }
        
        # Apply transforms
        if self.transforms:
            img = self.transforms(img)
        
        return img, target
    
    def parse_annotation(self, anno_path):
        """Parse VOC XML annotation"""
        tree = ET.parse(anno_path)
        root = tree.getroot()
        
        boxes = []
        labels = []
        
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in self.class_to_idx:
                continue
            
            difficult = obj.find('difficult')
            if difficult is not None and int(difficult.text) == 1:
                continue
            
            label = self.class_to_idx[name]
            bbox = obj.find('bndbox')
            
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)
        
        return boxes, labels


def get_transform(train=True, img_size=(480, 480)):
    """Get image transforms"""
    transforms = []
    
    # Resize all images to the same size (H, W)
    transforms.append(T.Resize(img_size))  # works on PIL images
    
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]))
    return T.Compose(transforms)


def collate_fn(batch):
    """Custom collate function for batching"""
    return tuple(zip(*batch))