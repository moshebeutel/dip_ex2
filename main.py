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
                  'sobel_vertical_edges': Kernel(
                      mat=np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]], dtype=np.float32)),
                  'sobel_horizontal_edges': Kernel(
                      mat=np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]], dtype=np.float32))}


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
    clean_image_grad.show(title='Clean Image after Gradient filter')
    noisy_image_grad = conv2d(original_img=noisy_image, kernel_name='horizontal_grad_left')
    noisy_image_grad.show(title='Noisy Image after Gradient filter')


def question2bc(noisy_image: DigitalImage):
    """
    question 2 part b, c: Apply Gaussian Filter on the noisy image and then apply gradient filter on the result.
                          Show results.
    :param noisy_image: DigitalImage. Original noisy image.
    :return: None
    """
    noisy_image_after_gaussian = conv2d(original_img=noisy_image, kernel_name='gaussian_7X7_sigma_10')
    noisy_image_after_gaussian.show(title='Noisy Image after Gaussian Filter')
    noisy_image_after_gaussian_and_grad = conv2d(original_img=noisy_image_after_gaussian, kernel_name='horizontal_grad_right')
    noisy_image_after_gaussian_and_grad.show(title='Noisy Image after Gaussian and Gradient Filters')


def question2d(noisy_image: DigitalImage):
    """
    Apply a Sobel Filter on the noisy image. Show results.
    :param noisy_image:  DigitalImage. Original noisy image.
    :return:
    """
    noisy_image_after_sobel_vertical = conv2d(original_img=noisy_image, kernel_name='sobel_vertical_edges')
    noisy_image_after_sobel_vertical.show(title='Noisy Image after Sobel Vertical Edges Filter')
    noisy_image_after_sobel_vertical_and_horizontal = conv2d(original_img=noisy_image_after_sobel_vertical,
                                                             kernel_name='sobel_horizontal_edges')
    noisy_image_after_sobel_vertical_and_horizontal.show(title='Noisy Image after Sobel Vertical and Horizontal Filter')


def question3ac(image1: DigitalImage, image2: DigitalImage):
    """
    Question 3 subsections a,c: Calculate Fourier transform for the two images.
     Visualize the amplitude and the phase.
     The function fulfills the task of *both* subsections 'a' and 'c'.
    :param image1: DigitalImage.
    :param image2: DigitalImage.
    :return: None
    """
    image1.calc_amplitude_phase()
    fig1 = image1.disp_amplitude_phase()
    fig1.savefig('Q3a - Regular Image Amplitude Phase.png')

    # # Noisy Image 'I_n':
    image2.calc_amplitude_phase()
    fig2 = image2.disp_amplitude_phase()
    fig2.savefig('Q3a - Noisy Image Amplitude Phase.png')


def question3b(image1: DigitalImage, image2: DigitalImage):
    """
    Question 3 subsection b: Subtract the amplitude of the  Fourier transforms for the two images 'I' and 'I_n'.
     Visualize the amplitude of the result.
    :param image1: DigitalImage.
    :param image2: DigitalImage.
    :return: None
    """

    sub =  np.abs(image1.amp - image2.amp)
    fig = plt.figure(figsize=(8,4))
    ax = plt.gca()
    z = ax.imshow(20*np.log(sub), cmap='gray')
    cbar = fig.colorbar(z, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Amplitude (dB)', rotation=90)
    ax.set_title(f'Amplitude Subtraction')
    fig.savefig('Subsec2 - Amplitude Substraction.png')

    
def question3d(image1: DigitalImage, image2: DigitalImage):
    """
    Question 3 part d: Combine the 'chita.jpeg' and 'zebra.jpeg' images using
    the amplitude of 'chita.jpeg' with the phase of 'zebra.jpeg'.
    Visualize the combined image.
    :param image1: DigitalImage. Cheetah image.
    :param image2: DigitalImage. Zebra image.
    :return: None
    """
    img_chita = image1._img
    img_zebra = image2._img
    # Note - we can't use the original images as their dimension mismatch. We have to manually pad both images into
    #  larger images of the same size.
    m = max(img_chita.shape[0], img_zebra.shape[1])
    n = max(img_chita.shape[0], img_zebra.shape[1])

    img_chita_pad = np.zeros([m,n])
    img_zebra_pad = np.zeros([m,n])
    img_chita_pad[:img_chita.shape[0], :img_chita.shape[1]] = img_chita
    img_zebra_pad[:img_zebra.shape[0], :img_zebra.shape[1]] = img_zebra

    # Caclculating the Fourier transforms:
    zebra_fft = np.fft.fft2(img_zebra_pad)
    chita_fft = np.fft.fft2(img_chita_pad)
    # Combining:
    combined_fft = np.abs(chita_fft) * np.exp(1j * np.angle(zebra_fft))
    combined_img = np.fft.ifft2(combined_fft)
    fig = plt.figure()
    z = np.real(combined_img)  # We take the real value, as the imaginary parts are negligible.
    plt.imshow(z, cmap='gray')
    plt.title('Combined Zebra - Chita Image')
    fig.savefig('Subsec4 - Combined Image.png')
    
    
def main():
    # Question 2:
    clean_image = load_image(original_img_filename='I.jpg')
    clean_image.show(title='Original Clean Image')
    noisy_image = load_image(original_img_filename='I_n.jpg')
    noisy_image.show(title='Original Noisy Image')
    # question2a(clean_image=clean_image, noisy_image=noisy_image)
    # question2bc(noisy_image=noisy_image)
    question2d(noisy_image=noisy_image)
    
    # Question 3:
    question3ac(image1=clean_image, image2=noisy_image)
    question3b(image1=clean_image, image2=noisy_image)
    
    cheetah_image = load_image(original_img_filename='chita.jpeg')
    zebra_image = load_image(original_img_filename='zebra.jpeg')
    question3ac(image1=cheetah_image, image2=zebra_image)
    question3d(image1=cheetah_image, image2=zebra_image)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
