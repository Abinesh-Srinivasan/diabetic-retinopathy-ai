from vit_keras import vit
from keras.layers import Dense
from keras.models import Model


def build_vit(num_classes=5):
    base_model = vit.vit_b16(
        image_size=224, pretrained=True, include_top=False, pretrained_top=False
    )

    output = Dense(num_classes, activation="softmax")(base_model.output)
    model = Model(inputs=base_model.input, outputs=output)

    return model
