import os
import shutil
from tqdm import tqdm
from src import config

def scan_original_dataset():
    """
    Scans the original dataset and prints folder names and image counts.
    """
    print("=" * 60)
    print(f"Scanning original dataset at: {config.ORIGINAL_DATASET_DIR}")
    print("=" * 60)
    
    if not os.path.exists(config.ORIGINAL_DATASET_DIR):
        raise FileNotFoundError(f"Original dataset directory not found at: {config.ORIGINAL_DATASET_DIR}")
        
    splits = sorted(os.listdir(config.ORIGINAL_DATASET_DIR))
    total_found = 0
    for split in splits:
        split_path = os.path.join(config.ORIGINAL_DATASET_DIR, split)
        if os.path.isdir(split_path):
            print(f"Split: {split}")
            classes = sorted(os.listdir(split_path))
            for cls in classes:
                cls_path = os.path.join(split_path, cls)
                if os.path.isdir(cls_path):
                    num_files = len([f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))])
                    print(f"  Class: {cls} -> {num_files} images")
                    if cls in config.CLASSES:
                        total_found += num_files
    print(f"Total matching images found for target classes {config.CLASSES}: {total_found}")
    print("=" * 60)

def copy_selected_classes():
    """
    Copies images of target classes from the original 6-class dataset to the new folder.
    Appends the split name to each filename (e.g. train_xxx.png) to prevent naming collisions.
    """
    print(f"Creating new dataset directory: {config.NEW_DATASET_DIR}")
    
    # Create target class subfolders in the new root
    for cls in config.CLASSES:
        os.makedirs(os.path.join(config.NEW_DATASET_DIR, cls), exist_ok=True)
        
    print("Copying files to new folder. This may take a few minutes...")
    
    # Collect all copy tasks to run with tqdm progress bar
    copy_tasks = []
    splits = sorted(os.listdir(config.ORIGINAL_DATASET_DIR))
    for split in splits:
        split_path = os.path.join(config.ORIGINAL_DATASET_DIR, split)
        if os.path.isdir(split_path):
            for cls in config.CLASSES:
                cls_path = os.path.join(split_path, cls)
                if os.path.isdir(cls_path):
                    dest_cls_dir = os.path.join(config.NEW_DATASET_DIR, cls)
                    for filename in os.listdir(cls_path):
                        src_file = os.path.join(cls_path, filename)
                        if os.path.isfile(src_file):
                            # Prepend split to filename to avoid collisions
                            new_filename = f"{split}_{filename}"
                            dest_file = os.path.join(dest_cls_dir, new_filename)
                            copy_tasks.append((src_file, dest_file))
                            
    # Execute copy operations
    for src_file, dest_file in tqdm(copy_tasks, desc="Copying images", unit="file"):
        # Copy file if it doesn't already exist to allow resume/idempotence
        if not os.path.exists(dest_file):
            shutil.copy2(src_file, dest_file)
            
    print("\nCopy process completed.")
    print("=" * 60)
    print("Final image counts in the new 4-class folder:")
    for cls in config.CLASSES:
        cls_dir = os.path.join(config.NEW_DATASET_DIR, cls)
        count = len([f for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f))])
        print(f"  Class: {cls} -> {count} images")
    print("=" * 60)

if __name__ == "__main__":
    scan_original_dataset()
    copy_selected_classes()
