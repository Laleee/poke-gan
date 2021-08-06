import os
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt


def image_to_edge(image):
    # 1. convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. perform the canny edge detector to detect image edges
    # TODO: ovde ne znam kako tacno thresholdi uticu na ekstrakcije,
    #       ako bude trebalo da se vraca na to, vratiti se
    edges = cv2.Canny(gray, threshold1=30, threshold2=100)

    return ~edges


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
