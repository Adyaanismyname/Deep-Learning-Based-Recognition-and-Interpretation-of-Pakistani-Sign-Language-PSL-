#!/usr/bin/env python3
"""
Comprehensive diagnostic script to find the training/inference mismatch.
"""

import torch
import torch.nn.functional as F
from models.vgg_feature_extractor import VGGFeatureExtractor
from models.lstm import PSL_LSTM
from dataloader.dataset_prep_videos import PSLVideoDataset, val_transform, train_transform
import numpy as np

def diagnose_issue():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"🖥️  Device: {device}\n")
    
    # Load dataset
    dataset = PSLVideoDataset(
        root_dir='Dataset',
        transform=val_transform,  # ⚠️ CHECK THIS
        max_frames=30,
        sample_rate=2
    )
    
    print(f"📊 Dataset: {len(dataset)} videos")
    print(f"📋 Label map: {dataset.label_map}\n")
    
    # Load models
    print("🔧 Loading VGG extractor...")
    vgg_extractor = VGGFeatureExtractor('checkpoints/vgg16_psl_best.pth', device=device)
    
    print("🔧 Loading LSTM model...")
    checkpoint = torch.load('checkpoints/lstm_psl_best.pth', map_location=device, weights_only=False)
    lstm_model = PSL_LSTM(num_classes=4)
    lstm_model.load_state_dict(checkpoint['model_state_dict'])
    lstm_model.to(device)
    lstm_model.eval()
    
    print(f"✅ Models loaded")
    print(f"📊 Checkpoint info:")
    print(f"   - Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"   - Val Acc: {checkpoint.get('val_acc', 'N/A')}")
    print(f"   - Val Loss: {checkpoint.get('val_loss', 'N/A')}\n")
    
    # ==================================================================
    # DIAGNOSTIC 1: Check normalization/preprocessing
    # ==================================================================
    print("="*70)
    print("DIAGNOSTIC 1: Transform/Normalization Check")
    print("="*70)
    
    frames, label, path = dataset[0]
    print(f"📹 Sample video: {path}")
    print(f"🏷️  True label: {dataset.label_map[label]}")
    print(f"📊 Frames shape: {frames.shape}")
    print(f"📊 Frames dtype: {frames.dtype}")
    print(f"📊 Frames range: [{frames.min():.3f}, {frames.max():.3f}]")
    print(f"📊 Frames mean: {frames.mean():.3f}")
    print(f"📊 Frames std: {frames.std():.3f}")
    
    # Check if normalization looks correct (ImageNet stats)
    # After ImageNet normalization: mean ≈ 0, std ≈ 1
    if abs(frames.mean()) > 1.0 or frames.std() < 0.5:
        print("⚠️  WARNING: Normalization might be incorrect!")
        print("   Expected: mean ≈ 0, std ≈ 1 (ImageNet normalized)")
    else:
        print("✅ Normalization looks correct\n")
    
    # ==================================================================
    # DIAGNOSTIC 2: Feature extraction comparison
    # ==================================================================
    print("="*70)
    print("DIAGNOSTIC 2: Feature Extraction Analysis")
    print("="*70)
    
    with torch.no_grad():
        # Extract features
        features = vgg_extractor.extract_features(frames)
        print(f"📊 Features shape: {features.shape}")
        print(f"📊 Features dtype: {features.dtype}")
        print(f"📊 Features range: [{features.min():.3f}, {features.max():.3f}]")
        print(f"📊 Features mean: {features.mean():.3f}")
        print(f"📊 Features std: {features.std():.3f}")
        
        # Check for NaN or Inf
        if torch.isnan(features).any():
            print("❌ ERROR: Features contain NaN!")
        if torch.isinf(features).any():
            print("❌ ERROR: Features contain Inf!")
        
        # Check if features are all zeros or very small
        if features.abs().max() < 1e-5:
            print("⚠️  WARNING: Features are suspiciously small!")
    
    print()
    
    # ==================================================================
    # DIAGNOSTIC 3: Mask behavior check
    # ==================================================================
    print("="*70)
    print("DIAGNOSTIC 3: Mask Behavior Check")
    print("="*70)
    
    # Test with proper mask
    mask = torch.ones(30, device=device)
    features_batch = features.unsqueeze(0).to(device)
    mask_batch = mask.unsqueeze(0)
    
    with torch.no_grad():
        outputs = lstm_model(features_batch, mask=mask_batch)
        probs = F.softmax(outputs, dim=1)
        
    print(f"📊 Output logits: {outputs[0].cpu().numpy()}")
    print(f"📊 Probabilities: {probs[0].cpu().numpy()}")
    print(f"📊 Predicted class: {outputs.argmax(1).item()} ({dataset.label_map[outputs.argmax(1).item()]})")
    print(f"📊 True class: {label} ({dataset.label_map[label]})")
    
    # Check if logits are reasonable
    if outputs.abs().max() > 100:
        print("⚠️  WARNING: Logits are very large! Model might be overconfident.")
    
    print()
    
    # ==================================================================
    # DIAGNOSTIC 4: Compare with and without mask
    # ==================================================================
    print("="*70)
    print("DIAGNOSTIC 4: Mask Impact Test")
    print("="*70)
    
    with torch.no_grad():
        # With proper mask (all ones)
        outputs_masked = lstm_model(features_batch, mask=mask_batch)
        probs_masked = F.softmax(outputs_masked, dim=1)
        
        # Without mask (None)
        outputs_no_mask = lstm_model(features_batch, mask=None)
        probs_no_mask = F.softmax(outputs_no_mask, dim=1)
        
    print("With mask (all valid frames):")
    print(f"  Logits: {outputs_masked[0].cpu().numpy()}")
    print(f"  Probs: {probs_masked[0].cpu().numpy()}")
    print(f"  Pred: {dataset.label_map[outputs_masked.argmax(1).item()]}")
    
    print("\nWithout mask:")
    print(f"  Logits: {outputs_no_mask[0].cpu().numpy()}")
    print(f"  Probs: {probs_no_mask[0].cpu().numpy()}")
    print(f"  Pred: {dataset.label_map[outputs_no_mask.argmax(1).item()]}")
    
    diff = (probs_masked - probs_no_mask).abs().max().item()
    if diff > 0.01:
        print(f"\n⚠️  WARNING: Large difference with/without mask: {diff:.4f}")
    else:
        print(f"\n✅ Mask behavior consistent: diff={diff:.4f}")
    
    print()
    
    # ==================================================================
    # DIAGNOSTIC 5: Test multiple samples
    # ==================================================================
    print("="*70)
    print("DIAGNOSTIC 5: Multi-Sample Sanity Check")
    print("="*70)
    
    correct = 0
    total = min(20, len(dataset))  # Test first 20 samples
    
    for idx in range(total):
        frames, label, path = dataset[idx]
        
        with torch.no_grad():
            features = vgg_extractor.extract_features(frames)
            mask = torch.ones(30, device=device)
            
            features_batch = features.unsqueeze(0).to(device)
            mask_batch = mask.unsqueeze(0)
            
            outputs = lstm_model(features_batch, mask=mask_batch)
            predicted = outputs.argmax(1).item()
            
            is_correct = (predicted == label)
            correct += is_correct
            
            status = "✅" if is_correct else "❌"
            print(f"{status} Sample {idx}: True={dataset.label_map[label]:10s} "
                  f"Pred={dataset.label_map[predicted]:10s}")
    
    acc = 100 * correct / total
    print(f"\n📊 Accuracy on {total} samples: {acc:.1f}%")
    
    if acc < 50:
        print("❌ CRITICAL: Accuracy is very low! Major issue detected.")
    elif acc < 80:
        print("⚠️  WARNING: Accuracy lower than expected.")
    else:
        print("✅ Accuracy seems reasonable for quick test.")
    
    print()
    
    # ==================================================================
    # DIAGNOSTIC 6: Check train vs val transform
    # ==================================================================
    print("="*70)
    print("DIAGNOSTIC 6: Transform Comparison (CRITICAL)")
    print("="*70)
    
    # Load same video with train transform
    dataset_train = PSLVideoDataset(
        root_dir='Dataset',
        transform=train_transform,  # ⚠️ DIFFERENT TRANSFORM
        max_frames=30,
        sample_rate=2
    )
    
    frames_val, _, _ = dataset[0]
    frames_train, _, _ = dataset_train[0]
    
    print("Val transform stats:")
    print(f"  Shape: {frames_val.shape}")
    print(f"  Range: [{frames_val.min():.3f}, {frames_val.max():.3f}]")
    print(f"  Mean: {frames_val.mean():.3f}, Std: {frames_val.std():.3f}")
    
    print("\nTrain transform stats:")
    print(f"  Shape: {frames_train.shape}")
    print(f"  Range: [{frames_train.min():.3f}, {frames_train.max():.3f}]")
    print(f"  Mean: {frames_train.mean():.3f}, Std: {frames_train.std():.3f}")
    
    # Test with train transform
    with torch.no_grad():
        features_train = vgg_extractor.extract_features(frames_train)
        mask = torch.ones(30, device=device)
        features_batch_train = features_train.unsqueeze(0).to(device)
        mask_batch = mask.unsqueeze(0)
        outputs_train = lstm_model(features_batch_train, mask=mask_batch)
        probs_train = F.softmax(outputs_train, dim=1)
    
    print("\nPrediction with TRAIN transform:")
    print(f"  Probs: {probs_train[0].cpu().numpy()}")
    print(f"  Pred: {dataset.label_map[outputs_train.argmax(1).item()]}")
    
    print("\nPrediction with VAL transform:")
    print(f"  Probs: {probs_masked[0].cpu().numpy()}")
    print(f"  Pred: {dataset.label_map[outputs_masked.argmax(1).item()]}")
    
    if (probs_train[0] - probs_masked[0]).abs().max() > 0.1:
        print("\n❌ CRITICAL: Train and Val transforms produce different results!")
        print("   This is likely the issue. Check transform definitions.")
    else:
        print("\n✅ Train and Val transforms produce similar results")
    
    print()
    
    # ==================================================================
    # SUMMARY
    # ==================================================================
    print("="*70)
    print("DIAGNOSTIC SUMMARY")
    print("="*70)
    print("\n🔍 Key things to check:")
    print("1. Are you using train_transform or val_transform for inference?")
    print("2. Does VGG checkpoint match the one used during training?")
    print("3. Is the label_map ordering consistent?")
    print("4. Are normalization stats correct (ImageNet)?")
    print("\n💡 Most likely issues:")
    print("   - Wrong transform (train vs val)")
    print("   - Wrong VGG checkpoint")
    print("   - Label map mismatch")
    print("="*70)


if __name__ == '__main__':
    diagnose_issue()