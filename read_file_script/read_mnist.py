import numpy as np
import idx2numpy
import threading
from typing import List


def save_images(image_file: str, images: np.ndarray) -> None:
    n = len(images)
    with open(image_file, "w") as f:
        for i in range(n):
            flattened_array: np.ndarray = images[i].reshape(-1)
            for k in range(flattened_array.shape[0]):
                if k != flattened_array.shape[0] - 1:
                    f.write(f"{flattened_array[k]},")
                else:
                    f.write(f"{flattened_array[k]}")
            f.write(f"\n")


def save_labels(labels_file: str, labels: np.ndarray) -> None:
    with open(labels_file, "w") as f:
        for label in labels:
            f.write(f"{label}\n")


train_images_path: str = "../mnist_data/unzip_file/train-images-idx3-ubyte"
train_labels_path: str = "../mnist_data/unzip_file/train-labels-idx1-ubyte"
test_images_path: str = "../mnist_data/unzip_file/t10k-images-idx3-ubyte"
test_labels_path: str = "../mnist_data/unzip_file/t10k-labels-idx1-ubyte"

train_images_file: str = "train_images.csv"
train_labels_file: str = "train_labels.csv"
test_images_file: str = "test_images.csv"
test_labels_file: str = "test_labels.csv"

train_images: np.ndarray = idx2numpy.convert_from_file(train_images_path)
train_labels: np.ndarray = idx2numpy.convert_from_file(train_labels_path)
test_images: np.ndarray = idx2numpy.convert_from_file(test_images_path)
test_labels: np.ndarray = idx2numpy.convert_from_file(test_labels_path)

if __name__ == "__main__":
    works: List[threading.Thread] = []
    works.append(
        threading.Thread(target=save_images, args=(train_images_file, train_images))
    )
    works.append(
        threading.Thread(target=save_labels, args=(train_labels_file, train_labels))
    )
    works.append(
        threading.Thread(target=save_images, args=(test_images_file, test_images))
    )
    works.append(
        threading.Thread(target=save_labels, args=(test_labels_file, test_labels))
    )

    for work in works:
        work.start()

    for work in works:
        work.join()
