import os
import numpy as np
import idx2numpy
from PIL import Image

train_images_path: str = "../mnist_data/unzip_file/train-images-idx3-ubyte"
train_labels_path: str = "../mnist_data/unzip_file/train-labels-idx1-ubyte"

save_images_dir: str = "train_images/"
save_labels_file: str = "train_labels.txt"

os.mkdir(save_images_dir)

train_images: np.ndarray = idx2numpy.convert_from_file(train_images_path)
train_labels: np.ndarray = idx2numpy.convert_from_file(train_labels_path)

for i in range(len(train_images)):
    image: Image.Image = Image.fromarray(train_images[i], mode="L")
    image_path: str = os.path.join(save_images_dir, f"{i}.png")
    image.save(image_path)

with open(save_labels_file, "w") as f:
    for label in train_labels:
        f.write(f"{label}\n")
