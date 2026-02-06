import os
import numpy as np
from keras.utils import Sequence, to_categorical
from utils.preprocessing import preprocess_image


class FundusDataset(Sequence):
    """
    Custom data generator for retinal fundus images.
    """

    def __init__(self, root_dir, batch_size=16, num_classes=5, shuffle=True):
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.shuffle = shuffle

        self.image_paths = []
        self.labels = []

        for label in range(num_classes):
            class_dir = os.path.join(root_dir, str(label))
            if not os.path.exists(class_dir):
                continue

            for img_name in os.listdir(class_dir):
                self.image_paths.append(os.path.join(class_dir, img_name))
                self.labels.append(label)

        self.image_paths = np.array(self.image_paths)
        self.labels = np.array(self.labels)

        self.indices = np.arange(len(self.image_paths))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        batch_indices = self.indices[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]

        batch_images = []
        batch_labels = []

        for i in batch_indices:
            img = preprocess_image(self.image_paths[i])
            batch_images.append(img)
            batch_labels.append(self.labels[i])

        batch_images = np.array(batch_images)
        batch_labels = to_categorical(batch_labels, self.num_classes)

        return batch_images, batch_labels
