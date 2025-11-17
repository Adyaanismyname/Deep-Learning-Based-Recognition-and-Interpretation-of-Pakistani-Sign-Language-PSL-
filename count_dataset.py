"""
Dataset Sample Counter for PSL Augmented Videos
Counts the number of video samples in each class and provides detailed statistics.
"""

import os
from collections import defaultdict

def count_dataset_samples(dataset_path='Dataset_Augmented'):
    """
    Count video samples in the augmented dataset.
    
    Args:
        dataset_path: Path to the dataset directory
    
    Returns:
        dict: Statistics about the dataset
    """
    
    # Video file extensions to look for
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV'}
    
    class_counts = defaultdict(int)
    augmentation_counts = defaultdict(int)
    total_videos = 0
    
    print(f"🔍 Scanning dataset: {dataset_path}")
    print("=" * 60)
    
    if not os.path.exists(dataset_path):
        print(f"❌ ERROR: Dataset path '{dataset_path}' does not exist!")
        return None
    
    # Get all class directories
    class_dirs = []
    for item in sorted(os.listdir(dataset_path)):
        item_path = os.path.join(dataset_path, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            class_dirs.append(item)
    
    if not class_dirs:
        print(f"❌ No class directories found in {dataset_path}")
        return None
    
    print(f"📁 Found {len(class_dirs)} class directories")
    print()
    
    # Count samples for each class
    for class_name in class_dirs:
        class_path = os.path.join(dataset_path, class_name)
        
        # Get all video files in this class
        video_files = []
        for file in os.listdir(class_path):
            file_ext = os.path.splitext(file)[1]
            if file_ext in video_extensions:
                video_files.append(file)
        
        class_counts[class_name] = len(video_files)
        total_videos += len(video_files)
        
        # Count augmentation types
        augmentation_types = set()
        for video_file in video_files:
            # Extract augmentation type from filename
            # Format: sXXXX_augmentation_type.mp4
            if '_' in video_file:
                parts = video_file.split('_')
                if len(parts) >= 2:
                    aug_type = '_'.join(parts[1:]).replace('.mp4', '').replace('.avi', '').replace('.mov', '').replace('.mkv', '')
                    augmentation_types.add(aug_type)
                    augmentation_counts[aug_type] += 1
        
        print(f"📊 {class_name:15s}: {len(video_files):4d} videos")
        if len(video_files) > 0:
            print(f"    Augmentation types: {', '.join(sorted(augmentation_types))}")
        print()
    
    # Summary statistics
    print("=" * 60)
    print("📈 DATASET SUMMARY")
    print("=" * 60)
    print(f"Total Classes: {len(class_counts)}")
    print(f"Total Videos: {total_videos:,}")
    print(f"Average videos per class: {total_videos/len(class_counts):.1f}")
    print(f"Min videos per class: {min(class_counts.values())}")
    print(f"Max videos per class: {max(class_counts.values())}")
    print()
    
    # Augmentation breakdown
    if augmentation_counts:
        print("🔄 AUGMENTATION BREAKDOWN")
        print("-" * 40)
        for aug_type, count in sorted(augmentation_counts.items()):
            print(f"{aug_type:20s}: {count:4d} videos")
        print()
    
    # Classes with unusual counts
    avg_count = total_videos / len(class_counts)
    outlier_classes = []
    
    for class_name, count in class_counts.items():
        if count < avg_count * 0.5 or count > avg_count * 1.5:
            outlier_classes.append((class_name, count))
    
    if outlier_classes:
        print("⚠️  CLASSES WITH UNUSUAL COUNTS")
        print("-" * 40)
        for class_name, count in sorted(outlier_classes, key=lambda x: x[1]):
            status = "LOW" if count < avg_count else "HIGH"
            print(f"{class_name:15s}: {count:4d} videos ({status})")
        print()
    
    # Split preview (based on your training script ratios)
    train_split = 0.7
    val_split = 0.15
    test_split = 0.15
    
    train_size = int(total_videos * train_split)
    val_size = int(total_videos * val_split)
    test_size = total_videos - train_size - val_size
    
    print("🎯 TRAIN/VAL/TEST SPLIT PREVIEW")
    print("-" * 40)
    print(f"Train (70%): {train_size:,} videos")
    print(f"Val   (15%): {val_size:,} videos")
    print(f"Test  (15%): {test_size:,} videos")
    print(f"Total:       {total_videos:,} videos")
    print()
    
    print("✅ Dataset counting complete!")
    
    return {
        'total_videos': total_videos,
        'total_classes': len(class_counts),
        'class_counts': dict(class_counts),
        'augmentation_counts': dict(augmentation_counts),
        'train_size': train_size,
        'val_size': val_size,
        'test_size': test_size
    }


def save_report(stats, output_file='dataset_report.txt'):
    """Save the dataset statistics to a file"""
    if not stats:
        return
    
    with open(output_file, 'w') as f:
        f.write("PSL Dataset Augmented - Sample Count Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total Classes: {stats['total_classes']}\n")
        f.write(f"Total Videos: {stats['total_videos']:,}\n")
        f.write(f"Average per class: {stats['total_videos']/stats['total_classes']:.1f}\n\n")
        
        f.write("Class Distribution:\n")
        f.write("-" * 30 + "\n")
        for class_name, count in sorted(stats['class_counts'].items()):
            f.write(f"{class_name:15s}: {count:4d}\n")
        
        f.write(f"\nAugmentation Types:\n")
        f.write("-" * 30 + "\n")
        for aug_type, count in sorted(stats['augmentation_counts'].items()):
            f.write(f"{aug_type:20s}: {count:4d}\n")
        
        f.write(f"\nTrain/Val/Test Split:\n")
        f.write(f"Train: {stats['train_size']:,}\n")
        f.write(f"Val:   {stats['val_size']:,}\n")
        f.write(f"Test:  {stats['test_size']:,}\n")
    
    print(f"📄 Report saved to: {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Count samples in PSL augmented dataset')
    parser.add_argument('--dataset', type=str, default='Dataset_Augmented', 
                       help='Path to dataset directory')
    parser.add_argument('--save-report', action='store_true', 
                       help='Save detailed report to file')
    parser.add_argument('--output', type=str, default='dataset_report.txt',
                       help='Output file for report')
    
    args = parser.parse_args()
    
    # Count samples
    stats = count_dataset_samples(args.dataset)
    
    # Save report if requested
    if args.save_report and stats:
        save_report(stats, args.output)