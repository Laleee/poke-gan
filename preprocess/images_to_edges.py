import os
import shutil
import cv2
import numpy as np


def image_to_edge(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=30, threshold2=100)

    return ~edges


def pokemon_dataset_to_edges():
    filelist = []

    for root, dirs, files in os.walk('../data/pokemon_jpg/'):
        filelist = filelist + [os.path.join(root, x) for x in files if x.endswith(('.jpg', '.png'))]

    path = '../data/edge_jpg'

    if not os.path.exists(path):
        os.makedirs(path)
    else:
        # Removes all the subdirectories!
        shutil.rmtree(path)
        os.makedirs(path)

    for img in filelist:
        img_name = img.split("/")[-1]
        image = cv2.imread(img)
        edge_image = image_to_edge(image)

        cv2.imwrite(os.path.join(path, img_name), edge_image)


def image_to_approx_poly_dp(imageread):
    image_edge_poly = np.ones((imageread.shape[0],
                               imageread.shape[1],
                               3))

    # converting the input image to grayscale image using cvtColor() function
    imagegray = cv2.cvtColor(imageread,
                             cv2.COLOR_RGB2GRAY)

    threshold_param = 30
    # using threshold() function to convert the grayscale image to binary image
    _, imagethreshold = cv2.threshold(imagegray,
                                      threshold_param,
                                      255,
                                      cv2.THRESH_BINARY_INV)

    # finding the contours in the given image using findContours() function
    imagecontours, _ = cv2.findContours(imagethreshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour_sum = 0
    for contour in imagecontours:
        contour_sum += contour.sum()

    threshold_empirical = 250000
    while contour_sum < threshold_empirical:
        threshold_param += 30
        # using threshold() function to convert the grayscale image to binary image
        _, imagethreshold = cv2.threshold(imagegray,
                                          threshold_param,
                                          255,
                                          cv2.THRESH_BINARY_INV)

        # finding the contours in the given image using findContours() function
        imagecontours, _ = cv2.findContours(imagethreshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contour_sum = 0
        for contour in imagecontours:
            contour_sum += contour.sum()

    # for each of the contours detected, the shape of the contours is approximated using approxPolyDP() function
    # and the contours are drawn in the image using drawContours() function
    for count in imagecontours:
        epsilon = 0.01 * cv2.arcLength(count, True)
        approximations = cv2.approxPolyDP(count, epsilon, True)
        cv2.drawContours(image_edge_poly, [approximations], 0, 0, 2)

    return image_edge_poly


def pokemon_dataset_to_sketch():
    filelist = []

    for root, dirs, files in os.walk('../data/pokemon_jpg/'):
        filelist = filelist + [os.path.join(root, x) for x in files if x.endswith(('.jpg', '.png'))]

    path = '../data/sketch_jpg'

    if not os.path.exists(path):
        os.makedirs(path)
    else:
        # Removes all the subdirectories!
        shutil.rmtree(path)
        os.makedirs(path)

    for img in filelist:
        img_name = img.split("/")[-1]
        image = cv2.imread(img)

        sketch_image = image_to_approx_poly_dp(image)

        cv2.imwrite(os.path.join(path, img_name), 255 * sketch_image)
