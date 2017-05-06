import sys

import cv2

import panorama
import utils

if __name__ == '__main__':
    ROOT_DIR = sys.argv[1]
    focal_len = float(sys.argv[2])

    pano = panorama.stitch_panorama(utils.load_series(ROOT_DIR), focal_len)

    utils.show_image(pano)
    cv2.imwrite('result.jpg', pano)