from __future__ import annotations

import copy
import os.path
import numpy as np

from conv import Conv2d, Kernel
from digital_image import DigitalImage
from utils import gaussian_kernel

images_path = './images/'

KernelRegistry = {'horizontal_grad_right': Kernel(mat=np.array([-1.0, 0.0, 1.0] * 3, dtype=np.float32).reshape(3, 3)),
                  'horizontal_grad_left': Kernel(mat=np.array([1.0, 0.0, -1.0] * 3, dtype=np.float32).reshape(3, 3)),
                  'gaussian_7X7_sigma_10': Kernel(mat=gaussian_kernel(shape=(7, 7), sigma=2.0)),
                  'sobel': Kernel(
                      mat=np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]], dtype=np.float32))}


def conv2d(original_img: DigitalImage, kernel_name: str) -> DigitalImage:
    """
    Apply convolution to a DigitalImage object given a kernel name
    :param original_img: DigitalImage. Original image to apply the convolution on.
    :param kernel_name: str. Name of the required convolution. Should exist in the kernel registry
    :return: DigitalImage. Result of the convolution operation.
    """
    assert isinstance(original_img, DigitalImage),\
        f'original_image is expected to be DigitalImage. Got {type(original_img)}'
    img = copy.copy(original_img)
    k = KernelRegistry[kernel_name]
    img.convolve(conv=Conv2d(kernel=k))
    assert img.shape == original_img.shape, f'Expected conv2d not to chage image shape.' \
                                        f' Got original shape {original_img.shape}, shape after convolution {img.shape}'
    return img


def load_image(original_img_filename: str):
    """
    Load an image from file and initialize a DigitalImage object
    :param original_img_filename: str, Image filename
    :return: DigitalImage
    """
    img_path = os.path.join(images_path, original_img_filename)
    assert os.path.exists(img_path), f'Image file {img_path} does not exist'
    img = DigitalImage(img_path=img_path)
    return img


def question2a(clean_image: DigitalImage, noisy_image: DigitalImage):
    """
    Question 2 part a: Apply a simple Gradient Filter on the clean and the noisy images. Show results.
    :param clean_image: DigitalImage. Original clean image.
    :param noisy_image: DigitalImage. Original noisy image.
    :return: None
    """
    clean_image_grad = conv2d(original_img=clean_image, kernel_name='horizontal_grad_left')
    clean_image_grad.show()
    noisy_image_grad = conv2d(original_img=noisy_image, kernel_name='horizontal_grad_left')
    noisy_image_grad.show()


def question2bc(noisy_image: DigitalImage):
    """
    question 2 part b, c: Apply Gaussian Filter on the noisy image and then apply gradient filter on the result.
                          Show results.
    :param noisy_image: DigitalImage. Original noisy image.
    :return: None
    """
    noisy_image_after_gaussian = conv2d(original_img=noisy_image, kernel_name='gaussian_7X7_sigma_10')
    noisy_image_after_gaussian.show(title='Noisy Image after Gaussian Filter')
    noisy_image_after_gaussian_and_grad = conv2d(original_img=noisy_image, kernel_name='horizontal_grad_right')
    noisy_image_after_gaussian_and_grad.show(title='Noisy Image after Gaussian and Gradient Filters')


def question2d(noisy_image: DigitalImage):
    """
    Apply a Sobel Filter on the noisy image. Show results.
    :param noisy_image:  DigitalImage. Original noisy image.
    :return:
    """
    noisy_image_after_sobel = conv2d(original_img=noisy_image, kernel_name='sobel')
    noisy_image_after_sobel.show(title='Noisy Image after Sobel Filter')


def main():
    clean_image = load_image(original_img_filename='I.jpg')
    clean_image.show(title='Original Clean Image')
    noisy_image = load_image(original_img_filename='I_n.jpg')
    noisy_image.show(title='Original Noisy Image')
    question2a(clean_image=clean_image, noisy_image=noisy_image)
    question2bc(noisy_image=noisy_image)
    question2d(noisy_image=noisy_image)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
