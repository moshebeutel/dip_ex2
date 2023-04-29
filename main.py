import os.path

import numpy as np
import cv2 as cv

images_path = './images/'


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


class Conv2d:
    def __init__(self, kernel: Kernel):
        self._kernel = kernel

    def __call__(self, img: np.array):
        assert np.ndim(img) < 4, f'Convolution Expected an image at most 3d -  c,w,h or w,h. Got {np.ndim(img)}d'
        if np.ndim(img) == 2:
            img = img[np.newaxis, :]
        c, m, n = img.shape
        result_img = np.zeros_like(img)
        for ch in range(c):
            for i in range(1, m - 1):
                for j in range(1, n - 1):
                    result_img[ch, i, j] = self._kernel(img[ch, i - 1:i + 2, j - 1:j + 2])

        return result_img.mean(axis=0, keepdims=True)


def zero_pad(img: np.array):
    assert np.ndim(img) == 2, f'zero_pad expected 2d array for image. Got {np.ndim(img)}d'
    padded_image = np.zeros(shape=(img.shape[0] + 2, img.shape[1] + 2))
    assert padded_image[1:-1, 1:-1].shape == img.shape
    padded_image[1:-1, 1:-1] = np.copy(img)
    return padded_image


def main():
    for img_filename in ['I.jpg', 'I_n.jpg']:
        image_convolution_result, img, padded = convolution(img_filename)
        show_images(image_convolution_result, img, img_filename, padded)


def show_images(image_convolution_result, original_img, img_filename, padded_image):
    cv.imshow(f'Original Image {img_filename}', original_img)
    cv.waitKey(0)
    cv.imshow(f'Padded Image {img_filename}', padded_image)
    cv.waitKey(0)
    cv.imshow(f'Convolution Result {img_filename}', image_convolution_result[0])
    cv.waitKey(0)


def convolution(img_filename):
    img_path = os.path.join(images_path, img_filename)
    assert os.path.exists(img_path), f'Image file {img_path} does not exist'
    img = cv.imread(img_path, 0)
    img = np.array(img, dtype=np.float32) / 255.0
    print(f'Loaded image with shape {img.shape}')
    padded = zero_pad(img)
    conv = Conv2d(kernel=Kernel(mat=np.array([-1.0, 0, 1.0] * 3).reshape(3, 3)))
    print(f'Padded image has shape {padded.shape}')
    image_convolution_result = conv(padded)
    return image_convolution_result, img, padded


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
