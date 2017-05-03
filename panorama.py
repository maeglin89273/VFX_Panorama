import numpy as np
import cv2


def cylinder_projection(frag_imgs, focal_len):
    cylinder_h = frag_imgs[0].shape[0]
    cylinder_w = int(2 * np.arctan(frag_imgs[0].shape[1] / focal_len))
    result = []
    for frag_img in frag_imgs:
        cylinder_proj = np.zeros((cylinder_h, cylinder_w, frag_img.shape[2]))

        for y in range(cylinder_proj[0]):
            for x in range(cylinder_proj[1]):
                pass


    return result

def stitch_panorama(frag_imgs, focal_len):
    return cylinder_projection(frag_imgs, focal_len)
