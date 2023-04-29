import os.path
import numpy as np

from conv import Conv2d, Kernel
from digital_image import DigitalImage
from utils import gaussian_kernel

images_path = './images/'

KernelRegistry = {'horizontal_grad_right': Kernel(mat=np.array([-1.0, 0, 1.0] * 3).reshape(3, 3)),
                  'horizontal_grad_left': Kernel(mat=np.array([1.0, 0, -1.0] * 3).reshape(3, 3)),
                  'gaussian_7X7_sigma_10': Kernel(mat=gaussian_kernel(shape=(7, 7), sigma=10)),
                  'sobel': Kernel(mat=np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]))}


def convolution(img_filename, kernel_name: str | list[str]):
    img_path = os.path.join(images_path, img_filename)
    assert os.path.exists(img_path), f'Image file {img_path} does not exist'

    img = DigitalImage(img_path=img_path)
    img.show(title='Original_Image')

    if isinstance(kernel_name, str):
        kernel_name = [kernel_name]

    for kn in kernel_name:
        k = KernelRegistry[kn]
        img.convolve(conv=Conv2d(kernel=k))
        img.show(title=kn)


def question2a():
    for img_filename in ['I.jpg', 'I_n.jpg']:
        convolution(img_filename=img_filename, kernel_name='horizontal_grad_left')


def question2bc():
    convolution(img_filename='I_n.jpg', kernel_name=['gaussian_7X7_sigma_10', 'horizontal_grad_right'])


def question2d():
    convolution(img_filename='I_n.jpg', kernel_name='sobel')


def main():
    question2a()
    question2bc()
    question2d()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
