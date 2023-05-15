import numpy as np


class Kernel:
    """
        A class used to represent a convolution kernel.
        Used as a functor, class object, using the __call__ implementation.
        ...

        Attributes
        ----------
        _mat: np.array. The kernel matrix
        _k : int. Dimension of kernel. Assume rectangular.

        Methods
        -------
        __call__(target_mat: np.array)
                 Implement the use of Kernel object as a function on target_mat

        Properties
        __________
        shape: (int,int)
                The shape of the underlying matrix self._mat
    """
    def __init__(self, mat: np.array, normalize=True):
        """

        :param mat: np.array. The kernel matrix
        :param normalize: bool. True if dividing by matrix sum is desired.
        """
        assert np.ndim(mat) == 2, f'Class Kernel expected 2d array. Got {np.ndim(mat)}d'
        self._mat = mat

        if normalize:
            self._mat /= np.absolute(self._mat).sum()

        k, g = mat.shape
        assert k == g, f'Class Kernel expected  a KxK array. Got {k}X{g}'
        assert k % 2 == 1, f'Class Kernel expected  an odd sized kernel. Got {k}'
        self._k = k

    def __call__(self, target_mat: np.array) -> float:
        """
        Elementwise multiply the kernel with the target matrix
        :param target_mat: np.array. The target matrix to apply the multiplication on.
        :return:
        """
        assert np.ndim(target_mat) == 2, f'Class Kernel __call__ expected 2d array. Got {np.ndim(target_mat)}d'
        assert target_mat.shape == self._mat.shape, \
            f'Kernel expected {self._mat.shape} shaped array. Got {target_mat.shape}'

        res = np.multiply(self._mat, target_mat)
        res = float(res.mean())
        return res

    @property
    def shape(self) -> tuple[int, int]:
        """
        The shape of the underlying matrix self._mat
        :return: tuple[int]
        """
        return self._k, self._k


class Conv2d:
    """
     A class used to represent a 2D convolution operation.
     Used as a functor, class object, using the __call__ implementation.

      Attributes
        ----------
        _kernel - The underlying kernel
        _rows_half_sz - Number of rows below/above the central cell of kernel
        _cols_half_sz - Number of columns on the right/left of the central cell of kernel


     Properties
        __________
        shape: (int,int)
                The shape of the underlying kernel
    """
    def __init__(self, kernel: Kernel):
        self._kernel = kernel
        self._rows_half_sz = self.shape[0] // 2
        self._cols_half_sz = self.shape[1] // 2

    def __call__(self, img: np.array) -> np.array:
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
    def shape(self) -> tuple[int, int]:
        """
        The shape of the underlying kernel
        :return: tuple[int, int]
        """
        return self._kernel.shape
