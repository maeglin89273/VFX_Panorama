import numpy as np
import cv2


def cylinder_projection(frag_imgs, focal_len):
    cylinder_h = frag_imgs[0].shape[0]
    cylinder_w = int(2 * np.arctan(frag_imgs[0].shape[1] / focal_len))
    result = []
    for frag_img in frag_imgs:
        cylinder_proj = np.zeros((cylinder_h, cylinder_w, frag_img.shape[2]))

        cylinder_coordinate = np.array(np.unravel_index(np.arange(cylinder_proj.shape[0] * cylinder_proj.shape[1]), cylinder_proj.shape[:2])).T
        sampling_locations = np.zeros_like(cylinder_coordinate)
        sampling_locations[:, 0] = (np.tan(cylinder_coordinate[:0] / focal_len) * focal_len).astype('int')
        sampling_locations[:, 1] = (np.tan(cylinder_coordinate[:0] / focal_len) * focal_len).astype('int')
    return result

def stitch_panorama(frag_imgs, focal_len):
    return cylinder_projection(frag_imgs, focal_len)
