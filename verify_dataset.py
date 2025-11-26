import os

voc_root = './data/VOCdevkit/VOC2007'

print("Checking PASCAL VOC 2007 dataset...")
print(f"Looking in: {os.path.abspath(voc_root)}\n")

required_dirs = {
    'Annotations': 'XML annotation files',
    'ImageSets/Main': 'Train/val/test splits',
    'JPEGImages': 'Image files',
}

all_good = True
for dir_path, description in required_dirs.items():
    full_path = os.path.join(voc_root, dir_path)
    if os.path.exists(full_path):
        num_files = len([f for f in os.listdir(full_path) if not f.startswith('.')])
        print(f"✓ {dir_path}: {num_files} files ({description})")
    else:
        print(f"✗ {dir_path}: NOT FOUND")
        all_good = False

if all_good:
    # Count images in splits
    splits = ['train', 'val', 'trainval', 'test']
    print("\nImage counts:")
    for split in splits:
        split_file = os.path.join(voc_root, 'ImageSets', 'Main', f'{split}.txt')
        if os.path.exists(split_file):
            with open(split_file) as f:
                count = len(f.readlines())
            print(f"  {split}: {count} images")
    
    print("\n✓ Dataset is ready!")
else:
    print("\n✗ Dataset incomplete. Please download missing files.")