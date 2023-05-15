import numpy as np


def constant_pad(img: np.array,
                 constant: float = 0.0,
                 add_rows_before: int = 1,
                 add_rows_after: int = 1,
                 add_cols_before: int = 1,
                 add_cols_after: int = 1):
    """
    This function implements a padding of a 2D np.array that represent an image by a given constant number
    :param img: a 2D NumPy array that will be padded
    :param constant: a float value that will be used as the constant value for padding. Default is 0.0.
    :param add_rows_before: an integer value that specifies the number of rows to add after the image. Default is 1.
    :param add_rows_after: an integer value that specifies the number of rows to add after the image. Default is 1.
    :param add_cols_before: an integer value that specifies the number of columns to add before the image. Default is 1.
    :param add_cols_after: an integer value that specifies the number of columns to add after the image. Default is 1.
    :return: a 2D numpy array representing the padded image
    """
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
             add_cols_after: int = 1) -> np.array:
    """
    This function implements zero padding of the input image by simply calling constant_pad()
     with constant=0.0.
    :param img: a 2D NumPy array that will be padded
    :param add_rows_before: an integer value that specifies the number of rows to add before the image. Default is 1.
    :param add_rows_after: an integer value that specifies the number of rows to add after the image. Default is 1.
    :param add_cols_before: an integer value that specifies the number of columns to add before the image. Default is 1.
    :param add_cols_after: an integer value that specifies the number of columns to add after the image. Default is 1.
    :return:a 2D numpy array representing the padded image
    """

    padded_image = constant_pad(img,
                                add_rows_before=add_rows_before,
                                add_rows_after=add_rows_after,
                                add_cols_before=add_cols_before,
                                add_cols_after=add_cols_after)
    return padded_image


def gaussian_kernel(shape: tuple[int, int] = (3, 3), sigma: float = 1.0):
    """
    Overall, this function generates a 2D Gaussian kernel with a given shape and standard deviation
     that can be used in various image processing tasks, such as smoothing, blurring, and edge detection.
    :param shape: a tuple of two integers that represents the shape of the kernel. The default value is (3, 3).
    :param sigma: a float value that represents the standard deviation of the Gaussian function.
     The default value is 1.0.
    :return: np.array. A 2D gaussian kernel with required shape and standard deviation.
    """
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
