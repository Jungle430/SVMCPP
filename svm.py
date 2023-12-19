from math import exp
from random import randint
from typing import Callable, List, Tuple
import numpy as np


def dual_svm(
    iterator,
    kernel_function: Callable[[np.ndarray, np.ndarray], float],
    number: int,
    C: float = 1.0,
    max_iter: int = 100,
    tol: float = 0.001,
) -> Tuple[np.ndarray, float]:
    _x: List[List[float]] = []
    _y: List[int] = []
    for record in iterator:
        _x.append(record[0])
        _y.append(1 if record[1] == number else -1)
    x: np.ndarray = np.array(_x)
    y: np.ndarray = np.array(_y)
    m: int = x.shape[0]
    alpha: np.ndarray = np.zeros(shape=(m,), dtype=np.float64)
    b: float = 0.0
    kernel_matrix: np.ndarray = np.zeros(shape=(m, m), dtype=np.float64)
    for i in range(m):
        for j in range(m):
            kernel_matrix[i][j] = kernel_function(x[i], x[j])
    for _ in range(max_iter):
        for i in range(m):
            error_i: float = 0.0
            for k in range(m):
                error_i += alpha[k] * y[k] * kernel_matrix[i][k]
            error_i += b - y[i]

            if (y[i] * error_i < -tol and alpha[i] < C) or (
                y[i] * error_i > tol and alpha[i] > 0
            ):
                j: int = randint(0, m - 1)
                error_j: float = 0.0
                for k in range(m):
                    error_j += alpha[k] * y[k] * kernel_matrix[j][k]
                error_j += b - y[j]

                alpha_i_old: float = alpha[i]
                alpha_j_old: float = alpha[j]
                L: float = 0.0
                H: float = 0.0
                if y[i] != y[j]:
                    L = max(0.0, alpha[j] - alpha[i])
                    H = min(C, C + alpha[j] - alpha[i])
                else:
                    L = max(0.0, alpha[i] + alpha[j] - C)
                    H = min(C, alpha[i] + alpha[j])

                eta: float = (
                    2 * kernel_matrix[i][j] - kernel_matrix[i][i] - kernel_matrix[j][j]
                )

                if eta >= 0:
                    continue

                alpha[j] = alpha[j] - (y[j] * (error_i - error_j)) / eta
                alpha[j] = max(L, min(H, alpha[j]))
                alpha[i] = alpha[i] + y[i] * y[j] * (alpha_j_old - alpha[j])
                b1: float = (
                    b
                    - error_i
                    - y[i] * (alpha[i] - alpha_i_old) * kernel_matrix[i][i]
                    - y[j] * (alpha[j] - alpha_j_old) * kernel_matrix[i][j]
                )
                b2: float = (
                    b
                    - error_j
                    - y[i] * (alpha[i] - alpha_i_old) * kernel_matrix[i][j]
                    - y[j] * (alpha[j] - alpha_j_old) * kernel_matrix[j][j]
                )
                if 0 < alpha[i] and alpha[i] < C:
                    b = b1
                elif 0 < alpha[j] and alpha[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2
    return alpha, b


def rbf_kernel(x: np.ndarray, y: np.ndarray, gamma: float = 1.0) -> float:
    assert x.shape == y.shape
    distance: float = 0.0
    n: int = x.shape[0]
    for i in range(n):
        diff: float = x[i] - y[i]
        distance += diff**2
    return exp(-gamma * distance)
