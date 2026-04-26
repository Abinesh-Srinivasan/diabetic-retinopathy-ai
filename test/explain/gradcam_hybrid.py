import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tensorflow as tf
import numpy as np
import cv2

from models.hybrid_model import build_hybrid
from utils.preprocessing import preprocess_image
from utils.explainability import make_gradcam_heatmap, overlay_gradcam

# ---------------- CONFIG ---------------- #
IMAGE_PATH = "data/test/2/d9a475dfe59a.png"  # change this
WEIGHTS_PATH = "hybrid_best.h5"
LAST_CONV_LAYER = "top_conv"  # EfficientNetB3 last conv layer
# ---------------------------------------- #

# Load hybrid model architecture
model = build_hybrid(num_classes=5)
model.load_weights(WEIGHTS_PATH)

# Load and preprocess image
original_img = cv2.imread(IMAGE_PATH)
original_img = cv2.resize(original_img, (224, 224))

img_array = preprocess_image(IMAGE_PATH)
img_array = np.expand_dims(img_array, axis=0)

# Generate Grad-CAM heatmap
heatmap = make_gradcam_heatmap(
    img_array=img_array, model=model, last_conv_layer_name=LAST_CONV_LAYER
)

# Overlay heatmap on original image
output = overlay_gradcam(original_img, heatmap)

# Display
cv2.imshow("Hybrid Grad-CAM", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
