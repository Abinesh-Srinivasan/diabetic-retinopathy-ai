from keras.applications import EfficientNetB3
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model


def build_cnn(num_classes=5):
    base_model = EfficientNetB3(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
    )

    base_model.trainable = False  # freeze backbone

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation="relu")(x)
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)
    return model
