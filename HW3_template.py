import cv2
import numpy as np
import matplotlib.pyplot as plt


def calculate_matches(image1, image2):
    # Load the images
    img1 = cv2.imread(image1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2, cv2.IMREAD_GRAYSCALE)

    # Initialize the SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors for both images
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    # Initialize the FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Perform matching
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Apply Lowe's ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    return good_matches, keypoints1, keypoints2


def dlt(matrix1, matrix2):
    # TODO : Get two sets of (x,y) coordinates from matches between images and calculate the homography by using SVD. return homography
    # return homography 3X3 matrix
    pass


def RANSAC(coordinates1, coordinates2, threshold, max_iterations=1000):
    # TODO : Get two sets of coordinates, a threshold value and max_iteration value, 4 points will be randomly picked
    #  from coordinates1 and coordinates2 .
    #  Then iterations will be done (max_iteration times) until the "best homography"
    #  will be found (the one that gives maximal count of inliners),
    #  return the best homography and number of inliners

    # return homography matrix 3X3, inliers
    pass


# def stitch_images(image1, image2, homography):
#     # Get images height and width , turn them to 1X2 vectors
#     h1, w1 = cv2.imread(image1).shape[:2]
#     h2, w2 = cv2.imread(image2).shape[:2]
#     corners1 = np.array([[0, 0], [0, h1], [w1, h1], [w1, 0]], dtype=np.float32).reshape(-1, 1, 2)
#     corners2 = np.array([[0, 0], [0, h2], [w2, h2], [w2, 0]], dtype=np.float32).reshape(-1, 1, 2)
#     # Translate the homography to take under account the field of view of the second image
#     corners2_transformed = cv2.perspectiveTransform(corners2, np.linalg.inv(homography).astype(np.float32))
#     all_corners = np.concatenate((corners1, corners2_transformed), axis=0)
#     x, y, w, h = cv2.boundingRect(all_corners)
#
#     # Adjust the homography matrix to map from img2 to img1
#     H_adjusted = np.linalg.inv(homography)
#
#     # Warp the images
#     img1_warped = cv2.warpPerspective(cv2.imread(image1), np.eye(3), (w, h))
#     img2_warped = cv2.warpPerspective(cv2.imread(image2), H_adjusted, (w, h))
#
#     # Combine the warped images into a single output image
#     output = cv2.addWeighted(img1_warped, 0.5, img2_warped, 0.5, 0)
#     plt.imshow(img1_warped)
#     plt.figure()
#     plt.imshow(img2_warped)
#
#     # Create a mask for the overlapping region
#     mask1 = np.zeros((h, w), dtype=np.uint8)
#     cv2.fillPoly(mask1, [np.int32(corners1)], (255))
#     mask2 = np.zeros((h, w), dtype=np.uint8)
#     cv2.fillPoly(mask2, [np.int32(corners2_transformed)], (255))
#     overlap_mask = cv2.bitwise_and(mask1, mask2)
#     not_overlap_img2_mask = cv2.bitwise_and(cv2.bitwise_not(overlap_mask), mask2)
#
#     # Blend only the overlapping region
#
#     blended = cv2.addWeighted(img1_warped, 0.5, img2_warped, 0.5, 0)
#
#     # Copy img1_warped and img2_warped to blended using the overlap_mask
#     blended = cv2.bitwise_and(blended, blended, mask=overlap_mask)
#     blended += cv2.bitwise_and(img1_warped, img1_warped, mask=cv2.bitwise_not(overlap_mask))
#     blended += cv2.bitwise_and(img2_warped, img2_warped, mask=not_overlap_img2_mask)
#
#     plt.figure()
#     plt.imshow(blended, cmap='gray')
#     cv2.imwrite('panoramic_image.jpg', blended)


# Main
def main():
    # Make sure you use the right path
    image1 = 'C/home/moshe/GIT/dip_ex2/images/Hanging1.png'
    image2 = '/home/moshe/GIT/dip_ex2/images/Hanging2.png'

    # Calculate good matches between the images and obtain keypoints
    matches, keypoints1, keypoints2 = calculate_matches(image1, image2)

    # Extract coordinates of the keypoints
    coordinates1 = np.float32([keypoints1[match.queryIdx].pt for match in matches])
    coordinates2 = np.float32([keypoints2[match.trainIdx].pt for match in matches])

    # RANSAC to find the best homography
    # homography, inliers = RANSAC(coordinates1, coordinates2, threshold=10000)
    #
    # # Stitch the images together
    # stitch_images(image1, image2, homography)


if __name__ == '__main__':
    main()
