#!/usr/bin/env python3
"""
Webcam inference for the trained VGG16 classifier (36 classes).

Behavior:
- Captures frames from webcam
- Uses MediaPipe Hands for robust hand detection and landmark tracking
- If hand is found: crops ROI around hand landmarks, resizes to 224x224, normalizes with ImageNet mean/std
- Loads the full VGG16 classifier checkpoint and runs inference (36-way)
- Displays top-3 class names and probabilities

Usage:
python infer_vgg_classifier_webcam.py --checkpoint checkpoints/vgg16_psl_best.pth

Dependencies:
pip install torch torchvision timm opencv-python mediapipe

Notes:
- This script assumes checkpoints contain a 'model_state_dict' key. If your checkpoint
  uses a different structure, set --checkpoint to a compatible file or adjust the loader.
"""

import argparse
import time
from pathlib import Path
import json

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import timm
import mediapipe as mp

# Default class mapping (from training). Index -> name
DEFAULT_LABEL_MAP = {
    0: '1-Hay', 1: 'Ain', 2: 'Alif', 3: 'Bay', 4: 'Byeh', 5: 'Chay', 6: 'Cyeh', 7: 'Daal',
    8: 'Dal', 9: 'Dochahay', 10: 'Fay', 11: 'Gaaf', 12: 'Ghain', 13: 'Hamza', 14: 'Kaf',
    15: 'Khay', 16: 'Kiaf', 17: 'Lam', 18: 'Meem', 19: 'Nuun', 20: 'Nuungh', 21: 'Pay',
    22: 'Ray', 23: 'Say', 24: 'Seen', 25: 'Sheen', 26: 'Suad', 27: 'Taay', 28: 'Tay',
    29: 'Tuey', 30: 'Wao', 31: 'Zaal', 32: 'Zaey', 33: 'Zay', 34: 'Zuad', 35: 'Zuey'
}


def load_label_map_from_json(path):
    """Load label map from JSON file (list or dict format)."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            # convert dict to ordered list
            ordered = [data[str(i)] for i in range(len(data))]
            return ordered
        if isinstance(data, list):
            return data
    except Exception as e:
        print(f"Warning: could not load label map from JSON: {e}")
    return None


def build_model(checkpoint_path, device, num_classes=36, model_name='vgg16'):
    """Build VGG16 model and load checkpoint."""
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading checkpoint from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location=device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
            print("✅ Checkpoint loaded successfully!")
        else:
            # try to load as direct state_dict
            try:
                model.load_state_dict(ckpt)
                print("✅ Checkpoint loaded successfully!")
            except Exception as e:
                print(f"⚠️  Warning: couldn't load checkpoint: {e}")
    
    model = model.to(device).eval()
    return model


def extract_hand_bbox(frame_bgr, mp_hands, results=None):
    """
    Extract hand bounding box using MediaPipe Hands detection.
    Returns: (bbox, hand_landmarks, handedness) or (None, None, None)
    bbox format: (x, y, w, h)
    """
    if results is None:
        # Convert BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(frame_rgb)
    
    if not results.multi_hand_landmarks:
        return None, None, None
    
    H, W = frame_bgr.shape[:2]
    
    # If multiple hands detected, prefer right hand, otherwise use first
    selected_idx = 0
    selected_handedness = "Right"
    
    if results.multi_handedness and len(results.multi_handedness) > 1:
        for idx, hand_handedness in enumerate(results.multi_handedness):
            label = hand_handedness.classification[0].label
            if label == "Right":
                selected_idx = idx
                selected_handedness = label
                break
    elif results.multi_handedness:
        selected_handedness = results.multi_handedness[0].classification[0].label
    
    hand_landmarks = results.multi_hand_landmarks[selected_idx]
    
    # Get bounding box from landmarks
    x_coords = [lm.x for lm in hand_landmarks.landmark]
    y_coords = [lm.y for lm in hand_landmarks.landmark]
    
    x_min = int(min(x_coords) * W)
    x_max = int(max(x_coords) * W)
    y_min = int(min(y_coords) * H)
    y_max = int(max(y_coords) * H)
    
    # Add padding (20% on each side)
    pad_x = int((x_max - x_min) * 0.20)
    pad_y = int((y_max - y_min) * 0.20)
    
    x_min = max(0, x_min - pad_x)
    y_min = max(0, y_min - pad_y)
    x_max = min(W, x_max + pad_x)
    y_max = min(H, y_max + pad_y)
    
    bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
    
    return bbox, hand_landmarks, selected_handedness


def preprocess_roi(roi_rgb):
    """Preprocess ROI: resize to 224x224, normalize with ImageNet mean/std."""
    img = cv2.resize(roi_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    
    # HWC -> CHW
    img = img.transpose(2, 0, 1)
    tensor = torch.from_numpy(img).unsqueeze(0)
    return tensor


def main():
    parser = argparse.ArgumentParser(description="VGG16 PSL Classifier Webcam Inference")
    parser.add_argument('--checkpoint', type=str, default='/Users/adyaanahmed/Downloads/best_model.pth',
                        help='Path to checkpoint file')
    parser.add_argument('--camera-id', type=int, default=0,
                        help='Camera device ID')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/mps/cpu). Auto-detected if not specified.')
    parser.add_argument('--label-map-json', type=str, default=None,
                        help='Optional JSON file with label map (list or dict)')
    parser.add_argument('--topk', type=int, default=3,
                        help='Number of top predictions to display')
    parser.add_argument('--show-landmarks', action='store_true',
                        help='Show MediaPipe hand landmarks on detected hand')
    parser.add_argument('--min-detection-confidence', type=float, default=0.7,
                        help='Minimum confidence for MediaPipe hand detection')
    parser.add_argument('--min-tracking-confidence', type=float, default=0.5,
                        help='Minimum confidence for MediaPipe hand tracking')
    args = parser.parse_args()

    # determine device
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

    # load label map
    label_list = [DEFAULT_LABEL_MAP[i] for i in range(len(DEFAULT_LABEL_MAP))]
    
    # override with JSON if provided
    if args.label_map_json:
        lm = load_label_map_from_json(args.label_map_json)
        if lm:
            label_list = lm
            print(f"📋 Loaded label map from JSON: {len(label_list)} classes")

    # try to load label_map from checkpoint if present
    if args.checkpoint and Path(args.checkpoint).exists():
        try:
            ckpt = torch.load(args.checkpoint, map_location='cpu')
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

    # build model and load checkpoint
    model = build_model(args.checkpoint, device, num_classes=len(label_list), model_name='vgg16')

    print(f"🎯 Model: VGG16 with {len(label_list)} classes")
    print(f"📹 Opening camera {args.camera_id}...")

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        print('❌ Error: cannot open webcam')
        return

    print("✅ Camera opened successfully!")
    print("🤚 Initializing MediaPipe Hands...")
    
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence
    )
    
    print("✅ MediaPipe initialized!")
    print("Press 'q' to quit\n")

    fps = 0.0
    last_time = 0.0
    
    # Temporal smoothing for predictions
    prediction_buffer = []
    buffer_size = 5  # smooth over last 5 frames
    last_bbox = None
    bbox_alpha = 0.6  # smoothing factor for bbox (more responsive than before)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            display_frame = frame.copy()
            
            # Process with MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_results = hands.process(frame_rgb)

            # extract hand region with MediaPipe
            bbox, hand_landmarks, handedness = extract_hand_bbox(frame, hands, mp_results)
            
            # Temporal smoothing of bounding box
            if bbox is not None:
                if last_bbox is not None:
                    # Smooth bbox transition
                    x, y, w, h = bbox
                    lx, ly, lw, lh = last_bbox
                    x = int(bbox_alpha * lx + (1 - bbox_alpha) * x)
                    y = int(bbox_alpha * ly + (1 - bbox_alpha) * y)
                    w = int(bbox_alpha * lw + (1 - bbox_alpha) * w)
                    h = int(bbox_alpha * lh + (1 - bbox_alpha) * h)
                    bbox = (x, y, w, h)
                last_bbox = bbox
                
            if bbox is not None:
                x, y, w, h = bbox
                roi = frame[y:y+h, x:x+w]
            else:
                # fallback: use center square crop
                H, W = frame.shape[:2]
                side = min(H, W)
                cx, cy = W // 2, H // 2
                x0 = max(0, cx - side // 2)
                y0 = max(0, cy - side // 2)
                roi = frame[y0:y0+side, x0:x0+side]
                x, y, w, h = x0, y0, side, side

            # convert BGR to RGB for model
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            inp = preprocess_roi(roi_rgb).to(device)

            # inference
            with torch.no_grad():
                outputs = model(inp)
                probs = F.softmax(outputs, dim=1)
                
                # Add to prediction buffer for temporal smoothing
                prediction_buffer.append(probs.cpu())
                if len(prediction_buffer) > buffer_size:
                    prediction_buffer.pop(0)
                
                # Average predictions over buffer
                if len(prediction_buffer) > 1:
                    avg_probs = torch.mean(torch.stack(prediction_buffer), dim=0)
                else:
                    avg_probs = probs.cpu()
                
                topk = torch.topk(avg_probs, k=min(args.topk, avg_probs.size(1)), dim=1)
                top_indices = topk.indices[0].tolist()
                top_probs = topk.values[0].tolist()

            # map indices to class names
            predictions = []
            for idx, prob in zip(top_indices, top_probs):
                name = label_list[idx] if idx < len(label_list) else f"Class_{idx}"
                predictions.append((name, float(prob), idx))

            # draw bounding box
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Draw MediaPipe hand landmarks if requested
            if args.show_landmarks and hand_landmarks is not None:
                mp_drawing.draw_landmarks(
                    display_frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
            
            # draw corner markers for better visibility
            corner_len = 20
            thickness = 3
            # top-left
            cv2.line(display_frame, (x, y), (x + corner_len, y), (0, 255, 0), thickness)
            cv2.line(display_frame, (x, y), (x, y + corner_len), (0, 255, 0), thickness)
            # top-right
            cv2.line(display_frame, (x+w, y), (x+w - corner_len, y), (0, 255, 0), thickness)
            cv2.line(display_frame, (x+w, y), (x+w, y + corner_len), (0, 255, 0), thickness)
            # bottom-left
            cv2.line(display_frame, (x, y+h), (x + corner_len, y+h), (0, 255, 0), thickness)
            cv2.line(display_frame, (x, y+h), (x, y+h - corner_len), (0, 255, 0), thickness)
            # bottom-right
            cv2.line(display_frame, (x+w, y+h), (x+w - corner_len, y+h), (0, 255, 0), thickness)
            cv2.line(display_frame, (x+w, y+h), (x+w, y+h - corner_len), (0, 255, 0), thickness)
            
            # Display handedness indicator
            hand_label = f"[{handedness}]" if handedness else ""
            
            # display top-1 prediction prominently with background
            main_text = f"{hand_label} {predictions[0][0]}: {predictions[0][1]*100:.1f}%"
            
            # add top-2 and top-3 as smaller text
            extra_text = ''
            if len(predictions) > 1:
                extra = ' | '.join([f"{name}:{prob*100:.0f}%" for name, prob, _ in predictions[1:]])
                main_text = main_text + f"   ({extra})"
            
            # Draw text with background for better readability
            text_y = max(30, y-15)
            (text_w, text_h), baseline = cv2.getTextSize(main_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(display_frame, (x-5, text_y - text_h - 5), 
                         (x + text_w + 5, text_y + baseline + 5), (0, 0, 0), -1)
            cv2.putText(display_frame, main_text, (x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # calculate and display FPS
            current_time = time.time()
            if last_time > 0:
                frame_time = current_time - last_time
                current_fps = 1.0 / frame_time if frame_time > 0 else 0
                fps = 0.9 * fps + 0.1 * current_fps
            last_time = current_time
            
            # FPS with background
            fps_text = f"FPS: {fps:.1f}"
            (fps_w, fps_h), fps_baseline = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(display_frame, (5, 5), (15 + fps_w, 15 + fps_h + fps_baseline), (0, 0, 0), -1)
            cv2.putText(display_frame, fps_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 255, 50), 2)

            cv2.imshow('VGG16 PSL Inference - Press q to quit', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()
        print("\n👋 Inference stopped")


if __name__ == '__main__':
    main()
