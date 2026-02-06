import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
from vit_keras import vit, visualize

# ---------------- CONFIG ---------------- #
IMAGE_PATH = "data/test/3/e8d1c6c07cf2.png"  # real image
MODEL_WEIGHTS = "vit_best.h5"
IMAGE_SIZE = 224
# ---------------------------------------- #

# Build ViT model (same config as training)
model = vit.vit_b16(
    image_size=IMAGE_SIZE,
    pretrained=True,
    include_top=True,
    pretrained_top=False,
    classes=5,
)

# Load trained weights
model.load_weights(MODEL_WEIGHTS)

# Load image (RAW IMAGE, NOT PREPROCESSED)
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise ValueError("Image not found. Check IMAGE_PATH.")

img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

# Generate attention map (expects raw image)
attention_map = visualize.attention_map(model, img)

# Normalize attention map
attention_map = cv2.normalize(attention_map, None, 0, 255, cv2.NORM_MINMAX)
attention_map = attention_map.astype("uint8")

# Overlay attention map
heatmap = cv2.applyColorMap(attention_map, cv2.COLORMAP_JET)
output = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

# Display
cv2.imshow("ViT Attention Map", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
