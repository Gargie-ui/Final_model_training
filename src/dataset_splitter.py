import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from src import config

def build_and_split_dataset():
    """
    Builds a DataFrame of all copied images, performs a stratified 70/15/15 split,
    caps No_Finding in the Train split to 5000 images, saves the CSVs,
    and saves a class balance bar chart showing all splits side-by-side.
    """
    print("=" * 60)
    print("Performing stratified split & train-only No_Finding cap...")
    print("=" * 60)
    
    # 1. Collect all filepaths and labels
    data = []
    for cls in config.CLASSES:
        cls_dir = os.path.join(config.NEW_DATASET_DIR, cls)
        if not os.path.exists(cls_dir):
            raise FileNotFoundError(f"Class directory not found: {cls_dir}. Run dataset_creator.py first.")
            
        for filename in os.listdir(cls_dir):
            filepath = os.path.abspath(os.path.join(cls_dir, filename))
            if os.path.isfile(filepath):
                data.append({"filepath": filepath, "label": cls})
                
    df = pd.DataFrame(data)
    if df.empty:
        raise ValueError(f"No images found in {config.NEW_DATASET_DIR}. Copy step failed or directory is empty.")
        
    print(f"Total images collected: {len(df)}")
    for cls in config.CLASSES:
        cls_count = len(df[df["label"] == cls])
        print(f"  {cls}: {cls_count}")
        
    # 2. Stratified split (70% train, 30% temp)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=config.RANDOM_SEED,
        stratify=df["label"]
    )
    
    # Stratified split temp (50% val, 50% test) to get 15% and 15% overall
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=config.RANDOM_SEED,
        stratify=temp_df["label"]
    )
    
    print("\nInitial stratified splits (before train downsampling):")
    print(f"  Train set size: {len(train_df)}")
    print(f"  Val set size: {len(val_df)}")
    print(f"  Test set size: {len(test_df)}")
    
    # 3. Downsample train split ONLY
    # Cap No_Finding to config.MAX_NO_FINDING_TRAIN_IMAGES (5000)
    train_no_finding = train_df[train_df["label"] == "No_Finding"]
    train_others = train_df[train_df["label"] != "No_Finding"]
    
    if len(train_no_finding) > config.MAX_NO_FINDING_TRAIN_IMAGES:
        print(f"\nDownsampling train 'No_Finding' from {len(train_no_finding)} to {config.MAX_NO_FINDING_TRAIN_IMAGES}...")
        train_no_finding = train_no_finding.sample(
            n=config.MAX_NO_FINDING_TRAIN_IMAGES,
            random_state=config.RANDOM_SEED
        )
    
    # Optional global cap for other classes in Train
    if config.MAX_IMAGES_PER_CLASS is not None:
        capped_others = []
        for cls in config.CLASSES:
            if cls == "No_Finding":
                continue
            cls_df = train_others[train_others["label"] == cls]
            if len(cls_df) > config.MAX_IMAGES_PER_CLASS:
                print(f"Downsampling train '{cls}' from {len(cls_df)} to {config.MAX_IMAGES_PER_CLASS}...")
                cls_df = cls_df.sample(n=config.MAX_IMAGES_PER_CLASS, random_state=config.RANDOM_SEED)
            capped_others.append(cls_df)
        train_others = pd.concat(capped_others)
        
    train_df = pd.concat([train_no_finding, train_others]).sample(frac=1.0, random_state=config.RANDOM_SEED).reset_index(drop=True)
    
    # Print final counts
    print("\nFinal splits after Train-only downsampling:")
    print(f"  Train set size: {len(train_df)}")
    print(f"  Val set size: {len(val_df)}")
    print(f"  Test set size: {len(test_df)}")
    
    print("\nFinal Class Counts per Split:")
    counts_dict = {"Class": config.CLASSES}
    for split_name, df_split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        counts = []
        for cls in config.CLASSES:
            count = len(df_split[df_split["label"] == cls])
            counts.append(count)
        counts_dict[split_name] = counts
        
    counts_df = pd.DataFrame(counts_dict)
    print(counts_df.to_string(index=False))
    
    # 4. Save CSVs
    os.makedirs(config.NEW_DATASET_DIR, exist_ok=True)
    train_df.to_csv(os.path.join(config.NEW_DATASET_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(config.NEW_DATASET_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(config.NEW_DATASET_DIR, "test.csv"), index=False)
    print(f"\nSaved train.csv, val.csv, test.csv in {config.NEW_DATASET_DIR}")
    
    # 5. Plot and save side-by-side bar chart
    plot_class_balance(counts_df)

def plot_class_balance(counts_df):
    """
    Plots a side-by-side bar chart of class counts in the Train, Val, and Test splits.
    """
    # Premium styles
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    classes = counts_df["Class"]
    train_counts = counts_df["Train"]
    val_counts = counts_df["Val"]
    test_counts = counts_df["Test"]
    
    x = np.arange(len(classes))
    width = 0.25
    
    # Harmonies colors
    rects1 = ax.bar(x - width, train_counts, width, label='Train (Capped)', color='#4F46E5', edgecolor='black', linewidth=0.5)
    rects2 = ax.bar(x, val_counts, width, label='Val (Uncapped)', color='#10B981', edgecolor='black', linewidth=0.5)
    rects3 = ax.bar(x + width, test_counts, width, label='Test (Uncapped)', color='#F59E0B', edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel('Image Count', fontsize=12, fontweight='bold')
    ax.set_title('Class Balance across Dataset Splits', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=10)
    ax.legend(frameon=True, facecolor='white', edgecolor='none')
    
    # Value annotations on top of the bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
                        
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    fig.tight_layout()
    chart_path = os.path.join(config.NEW_DATASET_DIR, "class_balance.png")
    plt.savefig(chart_path)
    plt.close()
    print(f"Saved class balance bar chart as: {chart_path}")
    print("=" * 60)

if __name__ == "__main__":
    build_and_split_dataset()
