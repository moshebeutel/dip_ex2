import numpy as np


class Kernel:

    def __init__(self, mat: np.array):
        assert np.ndim(mat) == 2, f'Class Kernel expected 2d array. Got {np.ndim(mat)}d'
        self._mat = mat
        k, g = mat.shape
        assert k == g, f'Class Kernel expected  a KxK array. Got {k}X{g}'
        assert k % 2 == 1, f'Class Kernel expected  an odd sized kernel. Got {k}'
        self._k = k

    def __call__(self, target_mat: np.array):
        assert np.ndim(target_mat) == 2, f'Class Kernel __call__ expected 2d array. Got {np.ndim(target_mat)}d'
        assert target_mat.shape == self._mat.shape, \
            f'Kernel expected {self._mat.shape} shaped array. Got {target_mat.shape}'
        return np.multiply(self._mat, target_mat).mean()

    @property
    def shape(self):
        return self._k, self._k


class Conv2d:
    def __init__(self, kernel: Kernel):
        self._kernel = kernel
        self._rows_half_sz = self.shape[0] // 2
        self._cols_half_sz = self.shape[1] // 2

    def __call__(self, img: np.array):
        assert np.ndim(img) < 4, f'Convolution Expected an image at most 3d -  c,w,h or w,h. Got {np.ndim(img)}d'
        if np.ndim(img) == 2:
            img = img[np.newaxis, :]
        c, m, n = img.shape
        result_img = np.zeros_like(img)
        for ch in range(c):
            for i in range(self._rows_half_sz, m - self._rows_half_sz):
                for j in range(self._cols_half_sz, n - self._cols_half_sz):
                    result_img[ch, i, j] = self._kernel(img[ch, i - self._rows_half_sz:i + self._rows_half_sz + 1,
                                                        j - self._cols_half_sz:j + self._cols_half_sz + 1])

        result_img = result_img.mean(axis=0, keepdims=True)

        print(f'Convolution finished with shape {result_img.shape} and dtype {result_img.dtype}')

        return result_img

    @property
    def shape(self):
        return self._kernel.shape
