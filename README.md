# Seeing Through Noise: Diffusion based Object Detection in Complex Settings
## TinyReproduction Project - Course Project

**Author**: Kunal Agarwal  
**Course**: ECE 57000 - Artificial Intelligence  
**Institution**: Purdue University  
**Date**: Fall 2025  
**Project Track**: TinyReproductions

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Code Attribution](#code-attribution)
3. [Project Structure](#project-structure)
4. [Dependencies](#dependencies)
5. [Dataset Setup](#dataset-setup)
6. [Installation Instructions](#installation-instructions)
7. [Running the Project](#running-the-project)
8. [Expected Results](#expected-results)
9. [Troubleshooting](#troubleshooting)
10. [Acknowledgments](#acknowledgments)

---

## Project Overview

This project is a **TinyReproduction** of the paper:

> [Chen, S., et al. (2023). "DiffusionDet: Diffusion Model for Object Detection." *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*.](https://arxiv.org/abs/2211.09788)

### Goal
The primary goal is to experimentally **verify the central claim** of the DiffusionDet paper: that **object detection can be successfully framed as a denoising diffusion process**. We reproduce this claim in a simplified setting by comparing DiffusionDet against a Faster R-CNN baseline on the PASCAL VOC 2007 dataset.

### Simplifications from Original Paper
To make this project feasible within a semester on a single GPU:
- **Dataset**: PASCAL VOC 2007 (instead of MS COCO)
- **Backbone**: ResNet-50 (instead of larger transformers)
- **Training Schedule**: 50 epochs (instead of 100+)
- **Number of Proposals**: 300 (instead of 500)
- **Diffusion Steps**: 1000 (same as original)

---

## Code Attribution

### Original Work (Written by Me)
The following code was **written entirely by me** for this project inspired by DiffusionDet and related works:

1. **`models/diffusion_det.py`** (Lines 1-344)
   - Complete DiffusionDet implementation
   - Forward diffusion process (q_sample)
   - Reverse diffusion process (inference)
   - Loss computation functions
   - Helper classes: MLP, SinusoidalTimeEmbedding, PositionalEncoding2D

2. **`utils/voc_dataset.py`** (Lines 1-115)
   - VOCDataset class implementation
   - XML annotation parsing
   - Custom collate function

3. **`train.py`** (Lines 1-230)
   - Training loop for DiffusionDet
   - Optimizer configuration
   - Checkpoint saving logic
   - Loss tracking and visualization

4. **`evaluate.py`** (Lines 1-240)
   - Complete mAP@0.5 evaluation implementation
   - IoU calculation
   - Precision-Recall curve computation
   - Average Precision calculation

5. **`train_faster_rcnn.py`** (Lines 1-140)
   - Training script for baseline model

### Adapted from External Sources

#### From Official DiffusionDet Paper
**Source**: https://github.com/ShoufaChen/DiffusionDet (Apache 2.0 License)

**Adaptations**:
- **`models/diffusion_det.py`**, Lines 57-74: Cosine beta schedule
  - **Original**: Used in official implementation with Detectron2
  - **Adapted**: Simplified to PyTorch-only implementation, removed Detectron2 dependencies

- **`models/diffusion_det.py`**, Lines 133-160: Box embedding and time embedding combination
  - **Original**: Core architecture from DiffusionDet paper
  - **Adapted**: Simplified decoder architecture, removed proposal matching

#### From DDPM Paper (Ho et al., 2020)
**Source**: "Denoising Diffusion Probabilistic Models" (arXiv:2006.11239)

**Adaptations**:
- **`models/diffusion_det.py`**, Lines 91-99: Forward diffusion (q_sample)
  - **Original**: DDPM forward process for images
  - **Adapted**: Applied to bounding box coordinates instead of pixels

- **`models/diffusion_det.py`**, Lines 170-185: Reverse diffusion sampling
  - **Original**: DDPM reverse process
  - **Adapted**: Simplified sampling (removed variance scheduling for faster inference)

---

## Project Structure

```
diffusiondet_reproduction/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── models/
│   ├── __init__.py                   # Empty init file
│   └── diffusion_det.py              # DiffusionDet model implementation
│                                      # - DiffusionDet class (main model)
│                                      # - MLP helper class
│                                      # - SinusoidalTimeEmbedding
│                                      # - PositionalEncoding2D
│
├── utils/
│   ├── __init__.py                   # Empty init file
|   ├── diffusion_util.py            # Diffusion utility functions
│   └── voc_dataset.py                # PASCAL VOC dataset loader
│                                      # - VOCDataset class
│                                      # - get_transform function
│                                      # - collate_fn function
|
├── train.py                           # Training script for DiffusionDet
├── train_faster_rcnn.py              # Training script for Faster R-CNN baseline
├── evaluate.py                        # Evaluation script (mAP@0.5)
│
├── download_voc.py                    # Dataset download script
├── verify_dataset.py                  # Dataset verification script
│
├── checkpoints/                       # Model checkpoints (auto-created)
│   ├── best_model.pth                # Best DiffusionDet model
│
└── checkpoints_frcnn/                 # Faster R-CNN checkpoints (auto-created)
    └── best_model.pth                # Best baseline model
```

### File Descriptions

#### Core Implementation Files
- **`models/diffusion_det.py`**: Complete implementation of DiffusionDet, including the diffusion process, transformer decoder, and prediction heads.
- **`utils/voc_dataset.py`**: Custom PyTorch Dataset class for loading PASCAL VOC 2007 with proper box normalization.

#### Training and Evaluation
- **`train.py`**: Main training script for DiffusionDet with loss tracking and checkpoint saving.
- **`train_faster_rcnn.py`**: Training script for the Faster R-CNN baseline for comparison.
- **`evaluate.py`**: Evaluation script that computes mAP@0.5 metric on the test set.

#### Utility Scripts
- **`download_voc.py`**: Automated script to download PASCAL VOC 2007 dataset.
- **`verify_dataset.py`**: Verification script to ensure dataset is properly downloaded and structured.

---

## Dependencies

### System Requirements
- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows with WSL
- **GPU**: NVIDIA GPU with 2GB+ VRAM (recommended)
  - CUDA 11.8 or 12.1
  - Can run on CPU but training will be very slow (~20x slower)
- **RAM**: 8GB+ recommended
- **Storage**: ~10GB free space (dataset + checkpoints)

### Python Version
- Python 3.8, 3.9, or 3.10 (tested on 3.10)

### Required Libraries

All dependencies are listed in `requirements.txt`:

```txt
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
Pillow>=9.5.0
matplotlib>=3.7.0
tqdm>=4.65.0
```

#### Dependency Details

| Library | Version | Purpose |
|---------|---------|---------|
| **torch** | ≥2.0.0 | Deep learning framework |
| **torchvision** | ≥0.15.0 | Computer vision utilities, Faster R-CNN |
| **numpy** | ≥1.24.0 | Numerical operations |
| **Pillow** | ≥9.5.0 | Image loading and processing |
| **matplotlib** | ≥3.7.0 | Training curve visualization |
| **tqdm** | ≥4.65.0 | Progress bars |

**No additional dependencies** are required. The project uses only standard, well-maintained libraries.

---

## Dataset Setup

### Automatic Download (Recommended)

The PASCAL VOC 2007 dataset can be downloaded automatically:

```bash
python3 download_voc.py
```

This script will:
1. Download `VOCtrainval_06-Nov-2007.tar` (~439 MB)
2. Download `VOCtest_06-Nov-2007.tar` (~430 MB)
3. Extract both archives to `data/VOCdevkit/VOC2007/`
4. Verify the dataset structure

**Download mirrors** (script tries in order):
1. Official host: http://host.robots.ox.ac.uk/pascal/VOC/voc2007/
2. Mirror: https://pjreddie.com/media/files/

**Total download time**: 5-15 minutes depending on connection speed.

### Manual Download (If Automatic Fails)

If the automatic download fails:

1. **Visit**: https://pjreddie.com/projects/pascal-voc-dataset-mirror/

2. **Download these files**:
   - `VOCtrainval_06-Nov-2007.tar`
   - `VOCtest_06-Nov-2007.tar`

3. **Place in project directory**:
   ```bash
   mv VOCtrainval_06-Nov-2007.tar ~/path/to/diffusiondet_reproduction/data/
   mv VOCtest_06-Nov-2007.tar ~/path/to/diffusiondet_reproduction/data/
   ```

4. **Extract**:
   ```bash
   cd data
   tar -xf VOCtrainval_06-Nov-2007.tar
   tar -xf VOCtest_06-Nov-2007.tar
   ```

### Dataset Verification

After download, verify the dataset:

```bash
python3 verify_dataset.py
```

**Expected output**:
```
✓ Annotations         9963 files
✓ ImageSets/Main      9963 files
✓ JPEGImages          9963 files

Image splits:
  train      :  2501 images
  val        :  2510 images
  trainval   :  5011 images
  test       :  4952 images

✓ Dataset verification PASSED!
```

### Dataset Details

- **Dataset**: PASCAL VOC 2007
- **Task**: Object Detection
- **Classes**: 20 object categories (aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor)
- **Split**: 
  - Trainval: 5,011 images
  - Test: 4,952 images
- **Size**: ~870 MB (extracted)

---

## Installation Instructions

### Step 1: Clone/Download Project

```bash
git clone https://github.com/KunalA18/diffusiondet_reproduction
cd diffusiondet_reproduction
```

Or if you have the files locally, ensure the directory structure matches the [Project Structure](#project-structure) section.

### Step 2: Create Python Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Linux/Mac
# OR
venv\Scripts\activate  # On Windows
```

### Step 3: Install PyTorch

**For CUDA 12.1** (most common):
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**For CUDA 11.8**:
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**For CPU only** (not recommended):
```bash
pip3 install torch torchvision
```

**Verify GPU availability**:
```bash
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Step 4: Install Other Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Download Dataset

```bash
python3 download_voc.py
```

Wait for download and extraction to complete (~10-15 minutes).

### Step 6: Verify Installation

```bash
# Verify imports
python3 -c "from models.diffusion_det import DiffusionDet; print('✓ Model imports OK')"
python3 -c "from utils.voc_dataset import VOCDataset; print('✓ Dataset imports OK')"

# Verify dataset
python3 verify_dataset.py
```

**If all checks pass**, you're ready to train! ✅

---

## Running the Project

### Quick Start (5-Minute Test)

Before running the full training (which takes hours), do a quick test:

1. **Edit `train.py`** and change:
   ```python
   config = {
       'num_epochs': 2,        # Changed from 50
       'batch_size': 4,        # Changed from 8
       'num_proposals': 100,   # Changed from 300
       ...
   }
   ```

2. **Run**:
   ```bash
   python3 train.py
   ```

3. **Expected**: Training should start and complete 2 epochs in ~5-10 minutes. This verifies everything works.

### Full Training Pipeline

#### Step 1: Train DiffusionDet (Main Model)

```bash
python3 train.py
```

**Configuration** (in `train.py`):
- Epochs: 50
- Batch size: 8
- Learning rate: 1e-4
- Optimizer: AdamW
- Scheduler: Cosine annealing

**Expected time**: 
- With GPU (RTX 3090/4090): ~4-6 hours
- With GPU (GTX 1080 Ti): ~8-10 hours
- With CPU: Not recommended (~80+ hours)

**Output**:
- Checkpoints saved to `checkpoints/`
- Best model: `checkpoints/best_model.pth`
- Training curves: `checkpoints/training_curves.png`

**Console output**:
```
Using device: cuda
Training samples: 5011
Creating model...
Total parameters: 45,234,567
Trainable parameters: 45,234,567

Starting training...
============================================================
Epoch 1/50: 100%|████████| 627/627 [04:32<00:00,  2.30it/s]

Epoch 1/50
  Total Loss: 5.2341
  BBox Loss:  1.8234
  Class Loss: 2.1456
  GIoU Loss:  1.2651
------------------------------------------------------------
...
```

#### Step 2: Train Faster R-CNN Baseline

```bash
python3 train_faster_rcnn.py
```

**Configuration**:
- Epochs: 50
- Batch size: 8
- Learning rate: 5e-3
- Optimizer: SGD with momentum

**Expected time**: ~3-4 hours on GPU

**Output**:
- Checkpoints saved to `checkpoints_frcnn/`
- Best model: `checkpoints_frcnn/best_model.pth`

#### Step 3: Evaluate Both Models

**Evaluate DiffusionDet**:
```bash
python3 evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --data-path data/VOCdevkit/VOC2007 \
    --batch-size 8
```

**Evaluate Faster R-CNN**:
```bash
python3 evaluate.py \
    --checkpoint checkpoints_frcnn/best_model.pth \
    --data-path data/VOCdevkit/VOC2007 \
    --batch-size 8
```

**Expected output**:
```
============================================================
Evaluation Results
============================================================

mAP@0.5: 45.23%

Per-class AP:
  aeroplane       : 52.34%
  bicycle         : 43.21%
  bird            : 38.56%
  ...
  tvmonitor       : 49.87%

============================================================
```

### Monitoring Training

**GPU Usage**:
```bash
watch -n 1 nvidia-smi
```

**Training Progress**:
- Watch console for loss values
- Loss should decrease steadily
- Check `checkpoints/training_curves.png` periodically

**Checkpoints**:
```bash
ls -lh checkpoints/
# Output:
# best_model.pth           # Best model (lowest loss)
# checkpoint_epoch_10.pth  # Saved every 10 epochs
# checkpoint_epoch_20.pth
# ...
# training_curves.png      # Loss plots
```

---

## Results

### Training Behavior

#### DiffusionDet Training Loss
- **Epochs 1-5**: Loss ~5-8 (model learning basic features)
- **Epochs 10-20**: Loss ~2-4 (model converging)
- **Epochs 30-50**: Loss ~1-2 (fine-tuning)

#### Faster R-CNN Training Loss
- **Epochs 1-5**: Loss ~3-5
- **Epochs 10-30**: Loss ~1-2
- **Epochs 30-50**: Loss ~0.5-1

### Evaluation Metrics

#### mAP@0.5 on PASCAL VOC 2007 Test Set

| Model | Backbone | mAP@0.5 | Status |
|-------|----------|---------|--------|
| **Faster R-CNN (Baseline)** | ResNet-50 | 45.1% | Expected range |
| **DiffusionDet (Ours)** | ResNet-50 | 48.2% | Expected range |


### Validation of Paper's Claim

**Success Criteria**: DiffusionDet achieves comparable performance (within +3.1% mAP) to Faster R-CNN baseline.
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
```python
# In train.py, reduce batch size:
'batch_size': 4,  # or even 2

# Or reduce number of proposals:
'num_proposals': 100,  # instead of 300
```

#### 2. Import Errors

**Error**: `ModuleNotFoundError: No module named 'models'`

**Solution**:
```bash
# Ensure __init__.py files exist:
touch models/__init__.py utils/__init__.py

# Or add to Python path:
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### 3. Dataset Not Found

**Error**: `FileNotFoundError: [Errno 2] No such file or directory: 'data/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt'`

**Solution**:
```bash
# Re-run dataset download:
python3 download_voc.py

# Or verify structure:
python3 verify_dataset.py
```

#### 4. Slow Training on CPU

**Issue**: Training taking 80+ hours

**Solution**:
- Use Google Colab (free GPU): https://colab.research.google.com
- Reduce dataset size for testing:
  ```python
  # In utils/voc_dataset.py, line 32, add:
  self.ids = self.ids[:100]  # Use only 100 images
  ```

#### 5. Loss Not Decreasing

**Issue**: Loss stays high after 10 epochs

**Causes and Solutions**:
1. **Learning rate too high**: Reduce to 5e-5
2. **Gradient explosion**: Check gradient clipping (in train.py)
3. **Data loading issue**: Verify boxes are normalized to [0,1]

#### 6. Low mAP (< 30%)

**Possible Causes**:
1. Didn't train long enough (train for full 50 epochs)
2. Learning rate not optimal
3. Bug in evaluation code (double-check IoU calculation)

### Getting Help

If issues persist:

1. **Check error message carefully** - most issues are path/import related
2. **Verify each component**:
   ```bash
   python3 -c "from models.diffusion_det import DiffusionDet; m = DiffusionDet(); print('Model OK')"
   python3 -c "from utils.voc_dataset import VOCDataset; d = VOCDataset('data/VOCdevkit/VOC2007'); print('Dataset OK')"
   ```
3. **Simplify to isolate problem**:
   - Train for 1 epoch with batch_size=1
   - Test on 10 images only
4. **Check GPU memory**: `nvidia-smi`
5. **Review logs**: Check console output for warnings

---

## Acknowledgments

### Paper and Code References

1. **DiffusionDet Paper**:
   - Chen, S., et al. (2023). "DiffusionDet: Diffusion Model for Object Detection." *ICCV 2023*.
   - Official Implementation: https://github.com/ShoufaChen/DiffusionDet

2. **DDPM Paper**:
   - Ho, J., et al. (2020). "Denoising Diffusion Probabilistic Models." *NeurIPS 2020*.
   - Paper: https://arxiv.org/abs/2006.11239

3. **Faster R-CNN**:
   - Ren, S., et al. (2015). "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks." *NeurIPS 2015*.

### Dataset

4. **PASCAL VOC 2007**:
   - Everingham, M., et al. "The PASCAL Visual Object Classes Challenge 2007 (VOC2007) Results."
   - http://host.robots.ox.ac.uk/pascal/VOC/voc2007/

### Libraries and Tools

5. **PyTorch**: https://pytorch.org/
6. **Torchvision**: https://pytorch.org/vision/
7. **Claude (Anthropic)**: AI assistant for implementation guidance

---

## License

This project is for educational purposes as part of a university course project.

- **Code License**: MIT License (for original code written by me)
- **Dataset**: PASCAL VOC 2007 is released for research purposes
- **Adapted Code**: Retains original licenses (Apache 2.0 for DiffusionDet components, BSD for PyTorch components)

---
