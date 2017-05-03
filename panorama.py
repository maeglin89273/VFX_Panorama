import numpy as np
import cv2
import utils


def cylinder_warp(frag_imgs, focal_len):

    result = []
    for frag_img in frag_imgs:
        cylinder_h = frag_img.shape[0]
        cylinder_w = int(focal_len * 2 * np.arctan(frag_img.shape[1] / (focal_len * 2)))
        cylinder_proj = np.zeros((cylinder_h, cylinder_w, frag_img.shape[2]), dtype=frag_img.dtype)
        half_img_w = frag_img.shape[1] / 2
        half_img_h = frag_img.shape[0] / 2
        half_cy_h = cylinder_proj.shape[0] / 2
        half_cy_w = cylinder_proj.shape[1] / 2

        focal_len_sq = focal_len ** 2
        for y in range(cylinder_proj.shape[0]):
            for x in range(cylinder_proj.shape[1]):
                theta = (x - half_cy_w) / focal_len
                c_img_x = focal_len * np.tan(theta)
                img_x = c_img_x + half_img_w
                img_y = half_img_h - (half_cy_h - y) * np.sqrt(focal_len_sq + c_img_x ** 2) / focal_len
                if img_y < frag_img.shape[0] and img_x < frag_img.shape[1]:
                    cylinder_proj[y, x, :] = frag_img[int(img_y), int(img_x), :]

        utils.show_images([cylinder_proj, frag_img])
        result.append(cylinder_proj)

    return result

def stitch_panorama(frag_imgs, focal_len):
    return cylinder_warp(frag_imgs, focal_len)
