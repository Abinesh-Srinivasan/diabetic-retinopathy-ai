import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import numpy as np
from keras.models import load_model
from sklearn.metrics import classification_report, accuracy_score

from utils.dataset import FundusDataset

val_data = FundusDataset("data/val", batch_size=16, num_classes=5, shuffle=False)
model = load_model("cnn_best.h5")

y_true, y_pred = [], []

for x, y in val_data:
    preds = model.predict(x, verbose=0)
    y_pred.extend(preds.argmax(axis=1))
    y_true.extend(y.argmax(axis=1))

print("CNN Validation Accuracy:", accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred))
