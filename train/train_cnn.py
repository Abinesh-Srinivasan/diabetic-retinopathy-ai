import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

from models.cnn_baseline import build_cnn
from utils.dataset import FundusDataset

NUM_CLASSES = 5
BATCH_SIZE = 16
EPOCHS = 25
LR = 1e-4

train_data = FundusDataset("data/train", BATCH_SIZE, NUM_CLASSES)
val_data = FundusDataset("data/val", BATCH_SIZE, NUM_CLASSES, shuffle=False)

model = build_cnn(NUM_CLASSES)

model.compile(optimizer=Adam(LR), loss="categorical_crossentropy", metrics=["accuracy"])

checkpoint = ModelCheckpoint("cnn_best.h5", monitor="val_accuracy", save_best_only=True)

early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop],
)
