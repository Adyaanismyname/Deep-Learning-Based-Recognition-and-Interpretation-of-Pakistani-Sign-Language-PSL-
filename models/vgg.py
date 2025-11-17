import timm
import torch.nn as nn

# Using TIMM for VGG16 (recommended)
MODEL_NAME = "vgg16"  # TIMM model name
NUM_CLASSES = 36

print(f"🔥 Loading {MODEL_NAME} from TIMM...")

# Load pretrained VGG16 from TIMM
model = timm.create_model(
    MODEL_NAME,
    pretrained=True,  # Load ImageNet pretrained weights
    num_classes=NUM_CLASSES,  # Modify final layer for your classes
)

print(f"✅ Model loaded successfully!")
print(f"📊 Model info: {model.num_classes} classes")
print(f"🎯 Input size: {model.default_cfg['input_size']}")

# Print model architecture
print(f"\n🏗️  Model Architecture:")
print(model)

# Fine-tuning: Freeze early layers, unfreeze last few layers
print(f"\n❄️  Freezing early layers for fine-tuning...")

# First, freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the classifier (final fully connected layers)
for param in model.head.parameters():
    param.requires_grad = True

# Unfreeze the last few convolutional blocks (last 2 blocks of VGG16)
# VGG16 has 5 blocks, we'll unfreeze blocks 4 and 5
unfrozen_layers = []
for name, param in model.named_parameters():
    # Unfreeze classifier
    if 'head' in name or 'classifier' in name:
        param.requires_grad = True
        unfrozen_layers.append(name)
    # Unfreeze last conv blocks (features.24 onwards is block 5, features.17-23 is block 4)
    elif 'features' in name:
        # Extract layer number
        layer_parts = name.split('.')
        if len(layer_parts) >= 2 and layer_parts[1].isdigit():
            layer_num = int(layer_parts[1])
            if layer_num >= 17:  # Unfreeze from block 4 onwards (layers 17+)
                param.requires_grad = True
                unfrozen_layers.append(name)

print(f"🔓 Unfrozen layers for fine-tuning:")
for layer in unfrozen_layers:
    print(f"   ✓ {layer}")

print(f"\n🔍 Trainable parameters:")
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"   Trainable: {trainable_params:,}")
print(f"   Total: {total_params:,}")
print(f"   Percentage: {100 * trainable_params / total_params:.1f}%")