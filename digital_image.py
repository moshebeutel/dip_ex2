import os
from math import sqrt

import cv2 as cv
import numpy as np

from conv import Conv2d
from utils import zero_pad
from matplotlib import pyplot as plt

class DigitalImage:
    """
    A class used to represent a Digital Image

     Attributes
        ----------
     _img: np.array. The underlying matrix
     _ndim: Number of dimensions
     _is_grayscale: True if the image is grayscale. False if RGB.
     _name: Filename.
     _phase: phase of fft
     _amp:   amplitude of fft


     Methods
     _______

     convolve(conv: Conv2d, log_mse: bool = True)
      convolve the image using Conv2d object. log the mse between source and result if needed

     show(title)
       show the image using imshow
     pad(pad_sizes)
        pad the image
     mse(other)
       compute mse between image and other image object

     Properties
     __________

    channels:
        Number of color channels. 1 for grayscale, 3 for RGB
    shape:
       Shape of the underlying matrix
    ndim:
     Number of image dimensions
    is_grayscale:
       True if the image is grayscale. False if colored.
    img:
       the underlying 2D array
    amp:
       amplitude of fft
    """
    def __init__(self, img_path: str):
        self._phase = None
        self._amp = None
        img = cv.imread(img_path, 0)
        print(f'Loaded image with shape {img.shape}')
        self._img = np.array(img, dtype=np.float32) / 255.0
        self._ndim = np.ndim(img)
        self._is_grayscale = self.ndim == 2 or self.ndim == 3 and self.shape[0] == 1
        self._name = os.path.basename(img_path)
        print('Image Name', self._name)

    def convolve(self, conv: Conv2d, log_mse: bool = True) -> None:
        """
        Apply a convolution on the image using a Conv2d object
        :param conv: The Conv2D object to be used for convolution
        :param log_mse: true if log the mse between result and source is needed
        :return: None
        """
        m, n = conv.shape
        assert m % 2 == 1, f'Expected odd sized kernel. Got {m}X{n}'
        assert n % 2 == 1, f'Expected odd sized kernel. Got {m}X{n}'
        rows_half_sz = m // 2
        cols_half_sz = n // 2
        padded = self.pad(pad_sizes=(rows_half_sz, rows_half_sz, cols_half_sz, cols_half_sz))
        img = conv(padded)[0][rows_half_sz:-rows_half_sz, cols_half_sz:-cols_half_sz]
        if log_mse:
            print('MSE after filter', self.mse(img))
        self._img = img

    def pad(self, pad_sizes: tuple[int, int, int, int] = (1, 1, 1, 1)) -> np.array:
        """
        Pad the image the sizes given
        :param pad_sizes: the sizes to pad
        :return: The padded matrix
        """
        return zero_pad(self._img, *pad_sizes)

    def show(self, title='') -> None:
        """
        Show the image
        :param title: Title of the image
        :return: None
        """
        sh = np.copy(self._img)
        sh = cv.normalize(sh, None, 255, 0, cv.NORM_MINMAX, cv.CV_8UC1)
        cv.imshow(title, sh)
        cv.waitKey(0)

    def mse(self, other) -> float:
        """
        Compute Mean Squared Error distance between image and other image
        :param other: The other DigitalImage object
        :return: mse result
        """
        assert self.shape == other.shape
        result = np.square(self._img - other).sum()
        result /= (self.shape[0] * self.shape[1])
        return sqrt(result)
    
    def calc_amplitude_phase(self):
        """
        Calculate the amplitude and phase of the image
        """
        # Fourier transform of the image:
        f = np.fft.fft2(self._img)
        # Centering frequencies:
        fshift = np.fft.fftshift(f)
        # Calculating the amplitude and phase:
        self._amp = np.abs(fshift)
        self._phase = np.angle(fshift)
        return
        
    def disp_amplitude_phase(self):
        """ Function that displays the amplitude and phase in a plot"""

        # We display the amplitude in decibels to better see the values.
        amp_dB = 20*np.log(np.abs(self._amp))
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        z = axes[0].imshow(amp_dB, cmap='gray')
        axes[0].set_title(f'Amplitude - Image \'{self._name}\'')
        axes[1].imshow(self._phase, cmap='gray')
        axes[1].set_title(f'Phase - Image \'{self._name}\'')
        cbar = fig.colorbar(z, ax=axes[0], fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel('Amplitude (dB)', rotation=90)
        fig.tight_layout(pad=8.0)
        plt.show()
        return fig

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

    @property
    def img(self):
        return self._img

    @property
    def amp(self):
        return self._amp
