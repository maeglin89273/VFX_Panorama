import os
import re

import cv2
from matplotlib import pyplot as plt

FRAG_IMG_NAME = re.compile('[a-zA-Z]+\d+')

def load_series(dir, scale=1.0):
    frag_img_filenames = [os.path.join(dir, filename) for filename in os.listdir(dir) if is_frag_img_name(filename)]
    tuples = [(os.path.splitext(os.path.basename(filename))[0], filename) for filename in frag_img_filenames]
    tuples.sort(key=lambda t: t[0])
    images = [cv2.resize(cv2.imread(filename), (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA) for t, filename in tuples]
    return images

def is_frag_img_name(filename):
    return FRAG_IMG_NAME.fullmatch(os.path.splitext(filename)[0])

def bgr_to_rgb(image):
    b, g, r = cv2.split(image)
    return cv2.merge([r, g, b])

def show_images(cv_images):
    cols = len(cv_images)
    for i, cv_image in enumerate(cv_images):
        plt.subplot(1, cols, i + 1)
        plt.imshow(bgr_to_rgb(cv_image))
        plt.xticks([]), plt.yticks([])

    plt.show()

def show_image(cv_image):
    plt.imshow(bgr_to_rgb(cv_image))
    plt.xticks([]), plt.yticks([])
    plt.show()

def show_heatmap(map, cmap='jet'):
    plt.imshow(map, cmap=cmap)
    plt.xticks([]), plt.yticks([])
    plt.show()

def plot_hist(xs, range, bins=20):
    plt.hist(xs, bins, range=range)
    plt.show()

def plot_hist2d(y, x, bins=20):
    plt.hist2d(x, y, bins, cmap='jet')
    plt.show()

def plot_img_with_feats(img, feat_locs):
    plt.imshow(img)
    plt.xticks([]), plt.yticks([])
    plt.scatter(feat_locs[:,1], feat_locs[:,0], c='r', s=2)
    plt.show()