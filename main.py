import os
import sys

import cv2

import panorama
import utils

if __name__ == '__main__':
    ROOT_DIR = sys.argv[1]
    focal_len = float(sys.argv[2])
    scale = 1 if len(sys.argv) <= 3 else float(sys.argv[3])
    pano = panorama.stitch_panorama(utils.load_series(ROOT_DIR, scale), focal_len)

    utils.show_image(pano)
    cv2.imwrite('%s_panorama.jpg' % os.path.basename(ROOT_DIR), pano)