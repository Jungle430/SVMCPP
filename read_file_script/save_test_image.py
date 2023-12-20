import os
from typing import Tuple
import idx2numpy
import numpy as np
import cv2

test_images_path: str = "../mnist_data/unzip_file/t10k-images-idx3-ubyte"
save_images_file: str = "test_images"
mnist_shape: Tuple[int, int] = (28, 28)
test_images: np.ndarray = idx2numpy.convert_from_file(test_images_path)


def clean_or_create_file(folder: str) -> bool:
    current_directory: str = os.getcwd()
    folder_path: str = os.path.join(current_directory, folder)
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path: str = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
                return False
    else:
        os.makedirs(folder_path)
    return True


def save_images(save_file: str, images: np.ndarray) -> None:
    for i, v in enumerate(images):
        image: np.ndarray = np.reshape(v, mnist_shape).astype(np.uint8)
        image_filename: str = os.path.join(save_file, f"mnist_image_{i}.png")
        cv2.imwrite(image_filename, image)


if __name__ == "__main__":
    ok: bool = clean_or_create_file(save_images_file)
    if not ok:
        print("Can't create the file!")
        exit(-1)
    save_images(save_images_file, test_images)
