import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)


import numpy as np
from sklearn.metrics import classification_report, accuracy_score

from models.vit_model import build_vit
from utils.dataset import FundusDataset
from utils.project_paths import project_path

NUM_CLASSES = 5
BATCH_SIZE = 16

val_data = FundusDataset(
    project_path("data", "val"), BATCH_SIZE, NUM_CLASSES, shuffle=False
)

model = build_vit(num_classes=NUM_CLASSES)
model.load_weights(project_path("vit_best.h5"))

y_true, y_pred = [], []

for x, y in val_data:
    preds = model.predict(x, verbose=0)
    y_pred.extend(preds.argmax(axis=1))
    y_true.extend(y.argmax(axis=1))

print("ViT Validation Accuracy:", accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred))
