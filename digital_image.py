import os
from math import sqrt

import cv2 as cv
import numpy as np

from conv import Conv2d
from utils import zero_pad


class DigitalImage:
    def __init__(self, img_path: str):
        img = cv.imread(img_path, 0)
        print(f'Loaded image with shape {img.shape}')
        self._img = np.array(img, dtype=np.float32) / 255.0
        self._ndim = np.ndim(img)
        self._is_grayscale = self.ndim == 2 or self.ndim == 3 and self.shape[0] == 1
        self._name = os.path.basename(img_path)
        print('Image Name', self._name)

    def convolve(self, conv: Conv2d, log_mse: bool = True):
        m, n = conv.shape
        assert m % 2 == 1, f'Expected odd sized kernel. Got {m}X{n}'
        assert n % 2 == 1, f'Expected odd sized kernel. Got {m}X{n}'
        rows_half_sz = m // 2
        cols_half_sz = n // 2
        print('rows_half_sz', rows_half_sz)
        print('cols_half_sz', cols_half_sz)
        padded = self.pad(pad_sizes=(rows_half_sz, rows_half_sz, cols_half_sz, cols_half_sz))
        print(f'Padded image has shape {padded.shape}')
        img = conv(padded)[0][rows_half_sz:-rows_half_sz, cols_half_sz:-cols_half_sz]
        if log_mse:
            print('MSE after filter', self.mse(img))
        self._img = img

    def pad(self, pad_sizes: tuple[int, int, int, int] = (1, 1, 1, 1)):
        return zero_pad(self._img, *pad_sizes)

    def show(self, title=''):
        sh = np.copy(self._img)
        cv.imshow(title + f'_{self._name}', sh)
        cv.waitKey(0)

    def mse(self, other):
        assert self.shape == other.shape
        result = np.square(self._img - other).sum()
        result /= (self.shape[0] * self.shape[1])
        return sqrt(result)

    @property
    def channels(self):
        return self._img.shape[0]

    @property
    def shape(self):
        return self._img.shape

    @property
    def ndim(self):
        return self._img.ndim

    @property
    def is_grayscale(self):
        return self._is_grayscale
