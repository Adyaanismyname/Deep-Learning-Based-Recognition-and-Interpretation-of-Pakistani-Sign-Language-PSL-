import cv2
import numpy as np
import random

# Read the image
img = cv2.imread("Dataset/1-Hay/s0251-10hay-Generated-169.jpg")
h, w = img.shape[:2]

# Random zoom-out factor (1 = no zoom, <1 = zoomed out)
zoom_factor = random.uniform(0.6, 0.9)

# Resize the original image (zoom out)
new_w, new_h = int(w * zoom_factor), int(h * zoom_factor)
resized = cv2.resize(img, (new_w, new_h))

# Create blurred background from original
blur_bg = cv2.GaussianBlur(img, (75, 75), 0)

# Paste the resized image in the center of the blurred background
x_offset = (w - new_w) // 2
y_offset = (h - new_h) // 2
zoomed_out = blur_bg.copy()
zoomed_out[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

# Show the images
cv2.imshow("Original", img)
cv2.imshow("Zoomed Out (centered on blurred background)", zoomed_out)
cv2.waitKey(0)
cv2.destroyAllWindows()
