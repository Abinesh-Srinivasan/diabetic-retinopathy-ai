from keras.applications import EfficientNetB3
from vit_keras import vit
from keras.layers import Dense, GlobalAveragePooling2D, Concatenate, Input
from keras.models import Model


def build_hybrid(num_classes=5):
    # -------- SHARED INPUT -------- #
    inputs = Input(shape=(224, 224, 3))

    # -------- CNN BRANCH -------- #
    cnn_base = EfficientNetB3(
        weights="imagenet", include_top=False, input_tensor=inputs
    )
    cnn_base.trainable = False

    cnn_features = GlobalAveragePooling2D()(cnn_base.output)

    # -------- ViT BRANCH -------- #
    vit_base = vit.vit_b16(
        image_size=224, pretrained=True, include_top=False, pretrained_top=False
    )

    vit_features = vit_base(inputs)  # 🔥 SAME INPUT

    # -------- FUSION -------- #
    fused = Concatenate()([cnn_features, vit_features])
    fused = Dense(512, activation="relu")(fused)
    fused = Dense(256, activation="relu")(fused)

    outputs = Dense(num_classes, activation="softmax")(fused)

    model = Model(inputs=inputs, outputs=outputs)
    return model
