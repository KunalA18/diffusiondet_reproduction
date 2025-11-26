"""
download_voc.py
Automatic Download Script for PASCAL VOC 2007 Dataset

This script downloads and extracts the PASCAL VOC 2007 dataset
for object detection. It tries multiple mirror sites and handles
network timeouts gracefully.

Usage:
    python3 download_voc.py

Output:
    data/VOCdevkit/VOC2007/ with complete dataset
"""

import os
import tarfile
import urllib.request
import urllib.error
import sys
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url, output_path, timeout=300):
    """
    Download file with progress bar and timeout handling
    
    Args:
        url: URL to download from
        output_path: Where to save the file
        timeout: Timeout in seconds (default 300 = 5 minutes)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"  Downloading from: {url}")
        with DownloadProgressBar(
            unit='B', 
            unit_scale=True,
            miniters=1, 
            desc=os.path.basename(output_path)
        ) as t:
            urllib.request.urlretrieve(
                url, 
                filename=output_path,
                reporthook=t.update_to,
                timeout=timeout
            )
        print(f"  ✓ Download complete: {output_path}")
        return True
    
    except urllib.error.URLError as e:
        print(f"  ✗ Download failed: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Unexpected error: {e}")
        return False


def extract_tar(tar_path, extract_path):
    """
    Extract tar file with progress
    
    Args:
        tar_path: Path to tar file
        extract_path: Where to extract
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"  Extracting: {os.path.basename(tar_path)}")
        file_size = os.path.getsize(tar_path)
        
        with tarfile.open(tar_path, 'r') as tar:
            members = tar.getmembers()
            with tqdm(total=len(members), desc="  Extracting files") as pbar:
                for member in members:
                    tar.extract(member, extract_path)
                    pbar.update(1)
        
        print(f"  ✓ Extracted to: {extract_path}")
        return True
    
    except Exception as e:
        print(f"  ✗ Extraction failed: {e}")
        return False


def verify_dataset(voc_root):
    """
    Verify that dataset was extracted correctly
    
    Args:
        voc_root: Path to VOC2007 directory
    
    Returns:
        bool: True if valid, False otherwise
    """
    required_dirs = [
        'Annotations',
        'ImageSets/Main',
        'JPEGImages',
        'SegmentationClass',
        'SegmentationObject'
    ]
    
    print("\n" + "="*70)
    print("Verifying dataset structure...")
    print("="*70)
    
    all_exist = True
    for dir_name in required_dirs:
        dir_path = os.path.join(voc_root, dir_name)
        if os.path.exists(dir_path):
            num_files = len([f for f in os.listdir(dir_path) 
                           if os.path.isfile(os.path.join(dir_path, f))])
            print(f"  ✓ {dir_name:<25} {num_files:>5} files")
        else:
            print(f"  ✗ {dir_name:<25} NOT FOUND")
            all_exist = False
    
    if all_exist:
        # Count images in splits
        print("\nImage splits:")
        splits = ['train', 'val', 'trainval', 'test']
        split_counts = {}
        
        for split in splits:
            split_file = os.path.join(voc_root, 'ImageSets', 'Main', f'{split}.txt')
            if os.path.exists(split_file):
                with open(split_file) as f:
                    count = len(f.readlines())
                split_counts[split] = count
                print(f"  {split:<10} : {count:>5} images")
        
        # Verify expected counts
        expected = {
            'train': 2501,
            'val': 2510,
            'trainval': 5011,
            'test': 4952
        }
        
        print("\nValidation:")
        all_correct = True
        for split, expected_count in expected.items():
            actual_count = split_counts.get(split, 0)
            if actual_count == expected_count:
                print(f"  ✓ {split:<10} : {actual_count} images (correct)")
            else:
                print(f"  ✗ {split:<10} : {actual_count} images (expected {expected_count})")
                all_correct = False
        
        return all_correct
    
    return False


def download_voc2007():
    """
    Main function to download and setup PASCAL VOC 2007
    """
    print("="*70)
    print("PASCAL VOC 2007 Dataset Download")
    print("="*70)
    print("\nThis will download and extract the PASCAL VOC 2007 dataset.")
    print("Total download size: ~870 MB")
    print("Estimated time: 10-20 minutes depending on connection speed")
    print()
    
    # Setup paths
    data_root = os.path.join(os.path.dirname(__file__), 'data')
    voc_root = os.path.join(data_root, 'VOCdevkit', 'VOC2007')
    
    os.makedirs(data_root, exist_ok=True)
    
    # Check if already downloaded
    if os.path.exists(voc_root):
        print(f"Dataset directory already exists: {voc_root}")
        print("Verifying existing dataset...")
        
        if verify_dataset(voc_root):
            print("\n" + "="*70)
            print("✓ Dataset already downloaded and verified!")
            print("="*70)
            print(f"\nDataset location: {os.path.abspath(voc_root)}")
            print("\nYou can proceed with training:")
            print("  python3 train.py")
            return
        else:
            print("\n⚠ Existing dataset is incomplete or corrupted.")
            response = input("Re-download? (y/n): ")
            if response.lower() != 'y':
                print("Exiting...")
                return
            print("\nRemoving incomplete dataset...")
            import shutil
            shutil.rmtree(os.path.join(data_root, 'VOCdevkit'))
    
    # Mirror URLs (try in order)
    mirrors = [
        {
            'name': 'Pjreddie Mirror (Recommended)',
            'trainval': 'https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar',
            'test': 'https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar'
        },
        {
            'name': 'Official PASCAL VOC Host',
            'trainval': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
            'test': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar'
        }
    ]
    
    # Files to download
    files_to_download = {
        'trainval': 'VOCtrainval_06-Nov-2007.tar',
        'test': 'VOCtest_06-Nov-2007.tar'
    }
    
    # Try each mirror
    download_success = False
    
    for mirror_idx, mirror in enumerate(mirrors, 1):
        print("\n" + "="*70)
        print(f"Attempting mirror {mirror_idx}/{len(mirrors)}: {mirror['name']}")
        print("="*70)
        
        all_downloaded = True
        
        for split, filename in files_to_download.items():
            filepath = os.path.join(data_root, filename)
            
            # Skip if already downloaded
            if os.path.exists(filepath):
                file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
                print(f"\n✓ {filename} already exists ({file_size_mb:.1f} MB)")
                print("  Skipping download...")
                continue
            
            # Download
            print(f"\nDownloading {split} set...")
            url = mirror[split]
            
            success = download_file(url, filepath, timeout=600)
            
            if not success:
                all_downloaded = False
                print(f"\n✗ Failed to download {filename} from {mirror['name']}")
                break
        
        if all_downloaded:
            download_success = True
            print(f"\n✓ Successfully downloaded all files from {mirror['name']}")
            break
    
    if not download_success:
        print("\n" + "="*70)
        print("❌ Automatic download failed from all mirrors")
        print("="*70)
        print("\nPlease download manually:")
        print("\n1. Visit: https://pjreddie.com/projects/pascal-voc-dataset-mirror/")
        print("\n2. Download these files:")
        print("   - VOCtrainval_06-Nov-2007.tar")
        print("   - VOCtest_06-Nov-2007.tar")
        print(f"\n3. Place them in: {os.path.abspath(data_root)}/")
        print("\n4. Run this script again to extract")
        sys.exit(1)
    
    # Extract files
    print("\n" + "="*70)
    print("Extracting archives...")
    print("="*70)
    
    for split, filename in files_to_download.items():
        filepath = os.path.join(data_root, filename)
        
        if not os.path.exists(filepath):
            print(f"\n✗ {filename} not found, skipping extraction")
            continue
        
        print(f"\n{split.upper()} set:")
        success = extract_tar(filepath, data_root)
        
        if not success:
            print(f"\n✗ Failed to extract {filename}")
            print("Please check the file and try again")
            sys.exit(1)
    
    # Verify dataset
    if verify_dataset(voc_root):
        print("\n" + "="*70)
        print("✅ Dataset setup complete!")
        print("="*70)
        print(f"\nDataset location: {os.path.abspath(voc_root)}")
        print("\nDataset statistics:")
        print("  - 20 object classes")
        print("  - 5,011 trainval images")
        print("  - 4,952 test images")
        print("  - 9,963 total images")
        
        # Cleanup option
        print("\n" + "="*70)
        print("Cleanup (Optional)")
        print("="*70)
        print("\nThe .tar files are no longer needed and take up ~870 MB.")
        response = input("Delete .tar files to save space? (y/n): ")
        
        if response.lower() == 'y':
            for filename in files_to_download.values():
                filepath = os.path.join(data_root, filename)
                if os.path.exists(filepath):
                    os.remove(filepath)
                    print(f"  ✓ Deleted: {filename}")
            print("\n✓ Cleanup complete!")
        
        print("\n" + "="*70)
        print("Next Steps:")
        print("="*70)
        print("\n1. Verify setup:")
        print("   python3 verify_dataset.py")
        print("\n2. Start training:")
        print("   python3 train.py")
        print("\n3. Or train baseline:")
        print("   python3 train_faster_rcnn.py")
        
    else:
        print("\n" + "="*70)
        print("❌ Dataset verification failed")
        print("="*70)
        print("\nSome files may be missing or corrupted.")
        print("Please try running the script again or download manually.")
        sys.exit(1)


if __name__ == '__main__':
    try:
        download_voc2007()
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user.")
        print("You can resume by running this script again.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        print("Please report this issue or try manual download.")
        sys.exit(1)