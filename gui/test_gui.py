from ast import literal_eval
from math import exp
import sys
from typing import Callable, List, Tuple

import pandas as pd
import idx2numpy
import numpy as np
import matplotlib.pyplot as plt
import cv2

svm_csv_file: str = "../model_data.csv"
base_image_path = "../read_file_script/test_images/mnist_image_"

test_images_path: str = "../mnist_data/unzip_file/t10k-images-idx3-ubyte"
test_labels_path: str = "../mnist_data/unzip_file/t10k-labels-idx1-ubyte"

train_images: np.ndarray = idx2numpy.convert_from_file(test_images_path)
train_labels: np.ndarray = idx2numpy.convert_from_file(test_labels_path)


def show(image: np.ndarray, number: int) -> None:
    _, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(image, cmap="gray")
    axes[0].axis("off")
    axes[1].text(0.5, 0.5, str(number), fontsize=35, ha="center", va="center")
    axes[1].axis("off")

    plt.show()


def alpha_b(csv_file: str) -> Tuple[np.ndarray, np.ndarray]:
    csv_data: pd.DataFrame = pd.read_csv(csv_file)
    csv_data_alpha = csv_data["alpha"].apply(literal_eval)
    csv_data_b = csv_data["b"].astype(float)
    alphas: np.ndarray = csv_data_alpha.values  # type: ignore
    bs: np.ndarray = csv_data_b.values  # type: ignore
    return alphas, bs


def getTrainMatrixImages(train_images: np.ndarray) -> np.ndarray:
    images_matrx: List[np.ndarray] = []
    for image in train_images:
        flattened_array: np.ndarray = image.reshape(-1)
        images_matrx.append(flattened_array)
    return np.array(images_matrx)


def getTrainMatrixLabels(labels: np.ndarray, number: int) -> np.ndarray:
    labels_matrix: List[int] = []
    for label in labels:
        labels_matrix.append(1 if int(label) == int(number) else -1)
    return np.array(labels_matrix)


def rbf_kernel(x: np.ndarray, y: np.ndarray, gamma: float = 1.0) -> float:
    assert x.shape == y.shape
    distance: float = 0.0
    n: int = x.shape[0]
    new_x = x.astype(np.float64)
    new_y = y.astype(np.float64)
    for i in range(n):
        diff: float = new_x[i] - new_y[i]
        distance += diff**2
    return exp(-gamma * distance)


def SVM_prediction(
    data_train: np.ndarray,
    label_train: np.ndarray,
    alpha: np.ndarray,
    b: float,
    kernel_function: Callable[[np.ndarray, np.ndarray], float],
    new_data: np.ndarray,
) -> float:
    m: int = min(len(data_train), len(alpha))
    predicted: float = b
    for j in range(m):
        predicted += (
            alpha[j] * label_train[j] * kernel_function(data_train[j], new_data)
        )
    return predicted


def SVM_prediction_number(
    data_train: np.ndarray,
    label_train: np.ndarray,
    alpha: np.ndarray,
    b: float,
    kernel_function: Callable[[np.ndarray, np.ndarray], float],
    new_data: np.ndarray,
    number: int,
) -> float:
    new_label_data: np.ndarray = getTrainMatrixLabels(label_train, number)
    return SVM_prediction(
        data_train, new_label_data, alpha, b, kernel_function, new_data
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("python3 test_gui.py <test_image number>")

    alphas, bs = alpha_b(svm_csv_file)
    image: np.ndarray = cv2.imread(
        f"{base_image_path}{sys.argv[1]}.png", cv2.IMREAD_GRAYSCALE
    )
    train_image_matrix: np.ndarray = getTrainMatrixImages(train_images)
    prediction_data: List[float] = [
        SVM_prediction_number(
            train_image_matrix,
            train_labels,
            np.array(alphas[i]),
            bs[i],
            rbf_kernel,
            image.reshape(-1),
            i,
        )
        for i in range(10)
    ]

    num: int = prediction_data.index(max(prediction_data))
    show(image, num)
