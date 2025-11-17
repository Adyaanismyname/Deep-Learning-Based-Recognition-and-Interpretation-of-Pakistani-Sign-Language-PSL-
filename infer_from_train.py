#!/usr/bin/env python3
"""
FULLY FIXED inference for LSTM PSL classifier.

KEY FIX: Handles variable-length sequences properly by extracting actual frame count
from the dataset and creating masks that match the real sequence length.
"""

import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from models.vgg_feature_extractor import VGGFeatureExtractor
from models.lstm import PSL_LSTM
from dataloader.dataset_prep_videos import PSLVideoDataset, val_transform
import cv2



def get_actual_frame_count(video_path, sample_rate=2):
    """Get the actual number of frames that will be sampled from video."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    # Calculate sampled frames
    sampled_frames = len(range(0, total_frames, sample_rate))
    return sampled_frames


def extract_features_with_proper_mask(frames, extractor, max_frames, device, actual_frame_count=None):
    """
    Extract features and create mask EXACTLY like training FeatureDataset.
    
    The key insight: The dataset may return padded frames (30 frames always),
    but we need to know the REAL frame count to create the correct mask.
    """
    with torch.no_grad():
        features = extractor.extract_features(frames)  # (seq_len, feature_dim)
    
    num_frames, feature_dim = features.shape
    
    # If actual_frame_count is provided, use it for masking
    # Otherwise use the feature count
    real_frame_count = actual_frame_count if actual_frame_count is not None else num_frames
    
    # Pad or truncate features to max_frames
    if num_frames < max_frames:
        padding = torch.zeros(max_frames - num_frames, feature_dim, device=features.device)
        features = torch.cat([features, padding], dim=0)
    elif num_frames > max_frames:
        features = features[:max_frames]
        real_frame_count = min(real_frame_count, max_frames)
    
    # Create mask based on REAL frame count
    mask = torch.ones(max_frames, device=features.device)
    if real_frame_count < max_frames:
        mask[real_frame_count:] = 0
    
    return features, mask, real_frame_count


def main():
    parser = argparse.ArgumentParser(description="FIXED LSTM PSL Classifier Inference")
    parser.add_argument('--data-dir', type=str, default='Dataset')
    parser.add_argument('--vgg-checkpoint', type=str, default='checkpoints/vgg16_psl_best.pth')
    parser.add_argument('--lstm-checkpoint', type=str, default='checkpoints/lstm_psl_best.pth')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--sample-idx', type=int, default=0)
    parser.add_argument('--max-frames', type=int, default=30)
    parser.add_argument('--sample-rate', type=int, default=2)
    parser.add_argument('--show-video', action='store_true')
    parser.add_argument('--infer-all', action='store_true')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='Number of samples to test (default: all)')
    args = parser.parse_args()

    # Device selection
    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')

    print(f"🖥️  Using device: {device}")

    # Load dataset
    print(f"📂 Loading dataset from: {args.data_dir}")
    dataset = PSLVideoDataset(
        root_dir=args.data_dir,
        transform=val_transform,
        max_frames=args.max_frames,
        sample_rate=args.sample_rate
    )
    
    if len(dataset) == 0:
        print("❌ Error: Dataset is empty!")
        return
    
    print(f"✅ Dataset loaded: {len(dataset)} videos")
    label_list = [dataset.label_map[i] for i in range(len(dataset.label_map))]
    print(f"📋 Classes: {label_list}\n")

    # Load models
    print("🔧 Loading VGG16 feature extractor...")
    extractor = VGGFeatureExtractor(args.vgg_checkpoint, device=device)
    print("✅ VGG16 loaded!")

    print("🔧 Loading LSTM model...")
    checkpoint = torch.load(args.lstm_checkpoint, map_location=device, weights_only=False)
    model = PSL_LSTM(num_classes=len(label_list))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"✅ LSTM loaded! (Epoch {checkpoint.get('epoch', 'N/A')}, "
          f"Val Acc: {checkpoint.get('val_acc', 0):.2f}%)\n")

    if args.infer_all:
        # Run inference on entire dataset
        num_samples = args.num_samples if args.num_samples else len(dataset)
        num_samples = min(num_samples, len(dataset))
        
        print(f"🔄 Running inference on {num_samples} samples...\n")
        
        correct = 0
        total = 0
        per_class_correct = {i: 0 for i in range(len(label_list))}
        per_class_total = {i: 0 for i in range(len(label_list))}
        
        for idx in range(num_samples):
            frames, label, video_path = dataset[idx]
            
            # Get actual frame count from video
            actual_frame_count = get_actual_frame_count(video_path, args.sample_rate)
            
            # Extract features with proper mask
            features, mask, real_frames = extract_features_with_proper_mask(
                frames, extractor, args.max_frames, device, actual_frame_count
            )
            
            # Add batch dimension
            features = features.unsqueeze(0)
            mask = mask.unsqueeze(0)
            
            with torch.no_grad():
                outputs = model(features, mask=mask)
                probs = F.softmax(outputs, dim=1)
                predicted = outputs.argmax(1).item()
                confidence = probs[0][predicted].item()
            
            is_correct = (predicted == label)
            correct += is_correct
            total += 1
            
            per_class_correct[label] += is_correct
            per_class_total[label] += 1
            
            status = "✅" if is_correct else "❌"
            true_label = label_list[label]
            pred_label = label_list[predicted]
            
            print(f"{status} [{idx+1:3d}/{num_samples}] "
                  f"True: {true_label:12s} | Pred: {pred_label:12s} ({confidence*100:5.1f}%) | "
                  f"Frames: {real_frames:2d}/{args.max_frames} | {Path(video_path).name}")
        
        # Summary
        accuracy = 100 * correct / total
        print(f"\n{'='*80}")
        print(f"RESULTS")
        print(f"{'='*80}")
        print(f"Overall Accuracy: {accuracy:.2f}% ({correct}/{total})")
        print(f"\nPer-Class Accuracy:")
        for i, class_name in enumerate(label_list):
            if per_class_total[i] > 0:
                class_acc = 100 * per_class_correct[i] / per_class_total[i]
                print(f"  {class_name:12s}: {class_acc:5.1f}% ({per_class_correct[i]}/{per_class_total[i]})")
        print(f"{'='*80}")
        
    else:
        # Single sample inference
        if args.sample_idx >= len(dataset):
            print(f"❌ Error: Sample index {args.sample_idx} out of range")
            return
        
        print(f"🎬 Loading sample {args.sample_idx}...")
        frames, label, video_path = dataset[args.sample_idx]
        
        print(f"📹 Video: {video_path}")
        print(f"🏷️  True Label: {label_list[label]}")
        print(f"📊 Frames shape: {frames.shape}")
        
        # Get actual frame count
        actual_frame_count = get_actual_frame_count(video_path, args.sample_rate)
        
        # Extract features with proper mask
        print("\n🔮 Running inference...")
        features, mask, real_frames = extract_features_with_proper_mask(
            frames, extractor, args.max_frames, device, actual_frame_count
        )
        
        print(f"📊 Features shape: {features.shape}")
        print(f"📊 Mask shape: {mask.shape}")
        print(f"📊 Valid frames: {real_frames}/{args.max_frames}")
        
        # Add batch dimension
        features = features.unsqueeze(0)
        mask = mask.unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(features, mask=mask)
            probs = F.softmax(outputs, dim=1)
            
            # Get top-3 predictions
            topk = torch.topk(probs, k=min(3, len(label_list)), dim=1)
            top_indices = topk.indices[0].tolist()
            top_probs = topk.values[0].tolist()
        
        print("\n" + "="*60)
        print("PREDICTION RESULTS")
        print("="*60)
        print(f"Video: {Path(video_path).name}")
        print(f"True Label: {label_list[label]}")
        print(f"Valid Frames: {real_frames}/{args.max_frames}")
        print(f"\nTop-3 Predictions:")
        for i, (idx, prob) in enumerate(zip(top_indices, top_probs), 1):
            pred_label = label_list[idx]
            status = "✅ CORRECT" if idx == label else ""
            print(f"  {i}. {pred_label}: {prob*100:.2f}% {status}")
        
        is_correct = top_indices[0] == label
        print(f"\nResult: {'✅ CORRECT' if is_correct else '❌ INCORRECT'}")
        print("="*60)
        
        # Show video if requested
        if args.show_video:
            print("\n🎥 Displaying video... (press 'q' to quit)")
            
            cap = cv2.VideoCapture(video_path)
            pred_label = label_list[top_indices[0]]
            pred_conf = top_probs[0]
            true_label = label_list[label]
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                display_frame = frame.copy()
                
                # Draw info
                texts = [
                    (f"Pred: {pred_label} ({pred_conf*100:.1f}%)", (0, 255, 0)),
                    (f"True: {true_label}", (255, 255, 0)),
                    ("CORRECT" if is_correct else "INCORRECT", (0, 255, 0) if is_correct else (0, 0, 255)),
                    (f"Valid Frames: {real_frames}/{args.max_frames}", (255, 255, 255))
                ]
                
                y_offset = 10
                for text, color in texts:
                    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(display_frame, (10, y_offset), 
                                (20 + text_w, y_offset + text_h + baseline + 10), (0, 0, 0), -1)
                    cv2.putText(display_frame, text, (15, y_offset + text_h + 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    y_offset += text_h + baseline + 15
                
                cv2.imshow('Video Playback', display_frame)
                
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()