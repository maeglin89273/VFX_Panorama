import sys

import panorama
import utils

if __name__ == '__main__':
    ROOT_DIR = sys.argv[1]
    focal_len = float(sys.argv[2])

    imgs = panorama.stitch_panorama(utils.load_series(ROOT_DIR), focal_len)

    print(imgs[0].shape)
    utils.show_images(imgs)