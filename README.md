# Pakistani Sign Language (PSL) Recognition System

A deep learning system for real-time Pakistani Sign Language recognition that combines static hand pose classification with dynamic gesture recognition.

## 🎯 Overview

This project implements a hybrid approach to sign language recognition:

- **Static Signs**: InceptionV3 CNN classifier for stationary hand poses (36 classes)
- **Dynamic Gestures**: LSTM with attention mechanism for movement-based signs (4 classes)
- **Intelligent Switching**: Automatic motion detection to switch between models in real-time

## 🏗️ Architecture

### Static Classifier (InceptionV3)

- Pre-trained InceptionV3 fine-tuned on PSL static signs
- MediaPipe hand detection for robust region of interest extraction
- 36 sign classes from the PSL alphabet

### Dynamic Classifier (LSTM + Attention)

- Bidirectional LSTM with attention mechanism
- InceptionV3 feature extractor for frame-level representations
- Temporal modeling for gesture sequences
- 4 dynamic gesture classes

### Motion Detection

- Frame differencing with hysteresis for state transitions
- Configurable patience thresholds to reduce noise
- Gaussian blur for robustness against camera jitter

## 📁 Project Structure

```
dl-semester-project/
├── infer_combined.py          # Main inference script (hybrid approach)
├── infer_vgg_classifier.py    # Static sign inference
├── infer_lstm_video.py        # Dynamic gesture inference
├── models/
│   ├── lstm.py                # LSTM model with attention
│   ├── vgg.py                 # InceptionV3 configuration
│   ├── vgg_feature_extractor.py
│   ├── extract_features.py    # Pre-extract features for training
│   ├── train_lstm.py          # LSTM training script
│   └── train_vgg.py           # InceptionV3 training script
├── dataloader/
│   ├── dataset_prep.py        # Image dataset utilities
│   └── dataset_prep_videos.py # Video dataset utilities
├── checkpoints/
│   ├── vgg16_psl_best.pth     # Trained InceptionV3 model
│   └── lstm_psl_best.pth      # Trained LSTM model
└── Dataset/                   # Training data
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Adyaanismyname/dl_semester_project.git
cd dl_semester_project

# Install dependencies
pip install torch torchvision timm opencv-python mediapipe albumentations tqdm
```

### Running Inference

**Combined Model (Recommended)**

```bash
# Webcam
python infer_combined.py

# Video file
python infer_combined.py --video-path path/to/video.mp4

# Adjust motion sensitivity
python infer_combined.py --motion-threshold 0.8
```

**Static Signs Only**

```bash
python infer_vgg_classifier.py
```

**Dynamic Gestures Only**

```bash
python infer_lstm_video.py
```

## 🎓 Training

### Training InceptionV3 (Static Classifier)

```bash
cd models
python train_vgg.py --dataset-path ../Dataset --epochs 50 --batch-size 32
```

### Training LSTM (Dynamic Classifier)

```bash
# Step 1: Extract InceptionV3 features (run once)
python extract_features.py

# Step 2: Train LSTM
python train_lstm.py --epochs 250
```

## 📊 Model Performance

### Static Classifier (InceptionV3)

- **Classes**: 36 PSL alphabet signs
- **Accuracy**: ~98.14% validation accuracy
- **Architecture**: InceptionV3 with fine-tuned layers

### Dynamic Classifier (LSTM)

- **Classes**: 4 dynamic gestures (2-Hay, Alifmad, Aray, Jeem)
- **Accuracy**: ~98.15% validation accuracy
- **Architecture**: 2-layer Bidirectional LSTM with attention

## 🎮 Controls

- Press **'q'** to quit
- Sentence builder automatically aggregates predictions
- Static signs require stable hand pose (~1 second)
- Dynamic gestures trigger on motion stop

## 🔧 Configuration

### Motion Detection Parameters

```python
motion_detector = MotionDetector(
    threshold=0.5,          # Motion sensitivity (lower = more sensitive)
    static_patience=5,      # Frames to wait before switching to static
    dynamic_patience=3      # Frames to wait before switching to dynamic
)
```

### Sentence Builder

```python
sentence_builder = SentenceBuilder()
# - Automatically debounces repeated predictions
# - Confidence thresholds: 0.6 for static, 0.5 for dynamic
# - Max 8 words in output sentence
```

## 📦 Dependencies

- Python 3.8+
- PyTorch 2.0+
- OpenCV
- MediaPipe
- Albumentations
- TIMM (PyTorch Image Models)
- tqdm

## 🙏 Acknowledgments

- Dataset: Pakistani Sign Language dataset
- Pre-trained weights: ImageNet (InceptionV3)
- Hand detection: MediaPipe Hands

## 📝 License

This project is for educational purposes.

## 👥 Authors

- Adyaan Ahmed

## 🐛 Known Issues

- Motion detection may trigger false positives in noisy environments
- MediaPipe hand detection requires good lighting
- LSTM requires minimum 12 frames for reliable prediction

## 🔮 Future Work

- [ ] Expand dynamic gesture vocabulary
- [ ] Real-time sentence translation
- [ ] Mobile deployment
- [ ] Multi-hand support
- [ ] Data augmentation improvements
