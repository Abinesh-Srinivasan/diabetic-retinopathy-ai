import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
from keras.models import load_model

from utils.preprocessing import preprocess_image
from utils.explainability import make_gradcam_heatmap, overlay_gradcam

# Load model
model = load_model("cnn_best.h5")

# IMPORTANT: last conv layer name (EfficientNetB3)
LAST_CONV_LAYER = "top_conv"

# Load image
img_path = "data/test/2/d9a475dfe59a.png"  # change to any test image
original = cv2.imread(img_path)
original = cv2.resize(original, (224, 224))

img_array = preprocess_image(img_path)
img_array = np.expand_dims(img_array, axis=0)

# Grad-CAM
heatmap = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER)
output = overlay_gradcam(original, heatmap)

cv2.imshow("Grad-CAM", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
