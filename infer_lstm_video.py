#!/usr/bin/env python3
"""
Webcam inference for LSTM-based PSL classifier with VGG16 feature extraction.

Features:
- Full frame processing (no hand detection)
- VGG16 feature extraction
- LSTM temporal modeling with frame sampling
- Smooth prediction display

Usage:
python infer_lstm_webcam.py
"""

import argparse
import time
from pathlib import Path
import json
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from models.vgg_feature_extractor import VGGFeatureExtractor
from models.lstm import PSL_LSTM
from dataloader.dataset_prep_videos import val_transform


DEFAULT_LABEL_MAP = {
    0: '2-Hay', 
    1: 'Alifmad', 
    2: 'Aray', 
    3: 'Jeem'
}


def load_label_map_from_json(path):
    """Load label map from JSON file (list or dict format)."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            ordered = [data[str(i)] for i in range(len(data))]
            return ordered
        if isinstance(data, list):
            return data
    except Exception as e:
        print(f"Warning: could not load label map from JSON: {e}")
    return None


def main():
    parser = argparse.ArgumentParser(description="LSTM PSL Classifier Webcam Inference")
    parser.add_argument('--vgg-checkpoint', type=str, default='checkpoints/vgg16_psl_best.pth',
                        help='Path to VGG16 checkpoint for feature extraction')
    parser.add_argument('--lstm-checkpoint', type=str, default='checkpoints/lstm_psl_best.pth',
                        help='Path to LSTM checkpoint')
    parser.add_argument('--camera-id', type=int, default=0,
                        help='Camera device ID')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/mps/cpu). Auto-detected if not specified.')
    parser.add_argument('--label-map-json', type=str, default=None,
                        help='Optional JSON file with label map')
    parser.add_argument('--max-frames', type=int, default=30,
                        help='Maximum number of frames for LSTM sequence')
    parser.add_argument('--sample-rate', type=int, default=2,
                        help='Sample every Nth frame (e.g., 2 = sample every 2nd frame)')
    args = parser.parse_args()

    # Determine device
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

    # Load label map
    label_list = [DEFAULT_LABEL_MAP[i] for i in range(len(DEFAULT_LABEL_MAP))]
    
    if args.label_map_json:
        lm = load_label_map_from_json(args.label_map_json)
        if lm:
            label_list = lm
            print(f"📋 Loaded label map from JSON: {len(label_list)} classes")

    # Try to load label_map from LSTM checkpoint
    if args.lstm_checkpoint and Path(args.lstm_checkpoint).exists():
        try:
            ckpt = torch.load(args.lstm_checkpoint, map_location='cpu', weights_only=False)
            if isinstance(ckpt, dict) and 'label_map' in ckpt:
                lm = ckpt['label_map']
                if isinstance(lm, dict):
                    ordered = [lm[i] for i in range(len(lm))]
                    label_list = ordered
                elif isinstance(lm, list):
                    label_list = lm
                print(f"📋 Loaded label_map from checkpoint: {len(label_list)} classes")
        except Exception:
            pass

    # Load VGG feature extractor
    print("🔧 Loading VGG16 feature extractor...")
    extractor = VGGFeatureExtractor(args.vgg_checkpoint)
    print("✅ VGG16 feature extractor loaded!")

    # Load LSTM model
    print("🔧 Loading LSTM model...")
    state = torch.load(args.lstm_checkpoint, map_location=device, weights_only=False)
    model = PSL_LSTM(num_classes=len(label_list))
    model.load_state_dict(state.get('model_state_dict'))
    model.to(device)
    model.eval()
    print("✅ LSTM model loaded!")

    print(f"🎯 Model: VGG16 + LSTM with {len(label_list)} classes")
    print(f"📹 Opening camera {args.camera_id}...")

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        print('❌ Error: cannot open webcam')
        return

    print("✅ Camera opened successfully!")
    print(f"📊 Collecting {args.max_frames} frames (sampling every {args.sample_rate} frame(s))")
    print("Press 'q' to quit\n")

    fps = 0.0
    last_time = 0.0
    frames = []  # sampled frames for inference
    frame_count = 0  # total frame counter for sampling
    current_prediction = None
    current_confidence = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            display_frame = frame.copy()
            frame_count += 1
            
            # Sample frames based on sample_rate (same as dataset logic)
            if frame_count % args.sample_rate == 0:
                # Process full frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Preprocess frame using val_transform
                transformed = val_transform(image=frame_rgb)
                frames.append(transformed['image'])  # torch tensor, CxHxW
                
                # When we have enough frames, run inference
                if len(frames) == args.max_frames:
                    with torch.no_grad():
                        frames_tensor = torch.stack(frames).float().to(device)  # (30, 3, 224, 224)
                        features = extractor.extract_features(frames_tensor)    # (30, feature_dim)
                        features = features.unsqueeze(0)                        # (1, 30, feature_dim)
                        outputs = model(features)
                        probs = F.softmax(outputs, dim=1)
                        
                        top_prob, top_idx = torch.max(probs, dim=1)
                        top_idx = top_idx.item()
                        top_prob = top_prob.item()
                        
                        current_prediction = label_list[top_idx] if top_idx < len(label_list) else f"Class_{top_idx}"
                        current_confidence = top_prob
                    
                    # Clear frames for next sequence
                    frames = []
                    frame_count = 0  # reset frame counter for next sequence

            # Display frame counter and prediction
            actual_frames = len(frames)
            frame_count_text = f"Sampled Frames: {actual_frames}/{args.max_frames}"
            total_needed = args.max_frames * args.sample_rate
            raw_frames_text = f"Raw Frames: {frame_count}/{total_needed}"
            
            if current_prediction is not None:
                pred_text = f"{current_prediction}: {current_confidence*100:.1f}%"
            else:
                pred_text = "Collecting frames..."
            
            # Draw prediction with background
            (text_w, text_h), baseline = cv2.getTextSize(pred_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cv2.rectangle(display_frame, (10, 50), 
                         (20 + text_w, 60 + text_h + baseline), (0, 0, 0), -1)
            cv2.putText(display_frame, pred_text, (15, 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            # Draw sampled frame counter
            (fc_w, fc_h), fc_baseline = cv2.getTextSize(frame_count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(display_frame, (10, 90), 
                         (20 + fc_w, 100 + fc_h + fc_baseline), (0, 0, 0), -1)
            cv2.putText(display_frame, frame_count_text, (15, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Draw raw frame counter
            (rf_w, rf_h), rf_baseline = cv2.getTextSize(raw_frames_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(display_frame, (10, 125), 
                         (20 + rf_w, 135 + rf_h + rf_baseline), (0, 0, 0), -1)
            cv2.putText(display_frame, raw_frames_text, (15, 145), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            # Calculate and display FPS
            current_time = time.time()
            if last_time > 0:
                frame_time = current_time - last_time
                current_fps = 1.0 / frame_time if frame_time > 0 else 0
                fps = 0.9 * fps + 0.1 * current_fps
            last_time = current_time
            
            fps_text = f"FPS: {fps:.1f}"
            (fps_w, fps_h), fps_baseline = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(display_frame, (5, 5), (15 + fps_w, 15 + fps_h + fps_baseline), (0, 0, 0), -1)
            cv2.putText(display_frame, fps_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 255, 50), 2)

            cv2.imshow('LSTM PSL Inference - Press q to quit', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n👋 Inference stopped")


if __name__ == '__main__':
    main()