import numpy as np


def constant_pad(img: np.array,
                 constant: float = 0.0,
                 add_rows_before: int = 1,
                 add_rows_after: int = 1,
                 add_cols_before: int = 1,
                 add_cols_after: int = 1):
    assert np.ndim(img) == 2, f'zero_pad expected 2d array for image. Got {np.ndim(img)}d'
    temp_mat_func = np.zeros if constant == 0.0 else np.ones
    padded_image = temp_mat_func(shape=(img.shape[0] + add_rows_before + add_rows_after,
                                        img.shape[1] + add_cols_before + add_cols_after), dtype=np.float32)
    if constant != 0.0:
        padded_image *= constant
    assert padded_image[add_rows_before:-add_rows_after, add_cols_before:-add_cols_after].shape == img.shape
    padded_image[add_rows_before:-add_rows_after, add_cols_before:-add_cols_after] = np.copy(img)
    return padded_image


def zero_pad(img: np.array,
             add_rows_before: int = 1,
             add_rows_after: int = 1,
             add_cols_before: int = 1,
             add_cols_after: int = 1):

    padded_image = constant_pad(img,
                                add_rows_before=add_rows_before,
                                add_rows_after=add_rows_after,
                                add_cols_before=add_cols_before,
                                add_cols_after=add_cols_after)
    return padded_image


def gaussian_kernel(shape: tuple[int, int] = (3, 3), sigma: float = 1.0):
    a = np.zeros(shape=shape, dtype=np.float32)
    m, n = shape
    half_rows = m // 2
    half_cols = n // 2
    for i in range(m):
        for j in range(n):
            r = float(i - half_rows)
            c = float(j - half_cols)
            a[i, j] = np.exp(-(r ** 2.0 + c ** 2.0) / (2.0 * sigma ** 2.0))
    a /= (2.0 * np.pi * sigma ** 2.0)

    return a
