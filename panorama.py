import numpy as np
import cv2
import scipy
from scipy import ndimage, spatial
import utils

def stitch_panorama(frag_imgs, focal_len):
    #cylinder_imgs, valid_maskes = cylinder_warp(frag_imgs, focal_len)
    descs_feat_locs = msop(frag_imgs)
    return stitch(frag_imgs, descs_feat_locs)


def cylinder_warp(frag_imgs, focal_len):
    valid_maskes = []
    cylinder_imgs = []
    for frag_img in frag_imgs:
        cylinder_h = frag_img.shape[0]
        cylinder_w = int(focal_len * 2 * np.arctan(frag_img.shape[1] / (focal_len * 2)))
        cylinder_proj = np.zeros((cylinder_h, cylinder_w, frag_img.shape[2]), dtype=frag_img.dtype)
        valid_mask = np.zeros(cylinder_proj.shape[:2], dtype=bool)
        half_img_h = frag_img.shape[0] / 2
        half_img_w = frag_img.shape[1] / 2
        half_cy_h = cylinder_h / 2
        half_cy_w = cylinder_w / 2

        focal_len_sq = focal_len ** 2
        for y in range(cylinder_proj.shape[0]):
            for x in range(cylinder_proj.shape[1]):
                theta = (x - half_cy_w) / focal_len
                c_img_x = focal_len * np.tan(theta)
                img_x = c_img_x + half_img_w
                img_y = half_img_h - (half_cy_h - y) * np.sqrt(focal_len_sq + c_img_x ** 2) / focal_len
                if 0 <= img_y < frag_img.shape[0] and 0 <= img_x < frag_img.shape[1]:
                    cylinder_proj[y, x, :] = frag_img[int(img_y), int(img_x), :]
                    valid_mask[y, x] = True

        cylinder_imgs.append(cylinder_proj)
        valid_maskes.append(valid_mask)

    return cylinder_imgs, valid_maskes



CORNER_THRESHOLD = 5000
MAX_FEAT_CANDIDATE_NUM = 3000
def msop(imgs):
    descs_feat_locs = [None] * len(imgs)
    for i, img in enumerate(imgs):
        print('msop on image %s' % i)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grad_I_x = cv2.GaussianBlur(cv2.Sobel(gray_img, cv2.CV_32F, 1, 0), (3, 3), 1)
        grad_I_y = cv2.GaussianBlur(cv2.Sobel(gray_img, cv2.CV_32F, 0, 1), (3, 3), 1)
        S_x_sq = cv2.GaussianBlur(grad_I_x * grad_I_x, (3, 3), 1.5)
        S_y_sq = cv2.GaussianBlur(grad_I_y * grad_I_y, (3, 3), 1.5)
        S_xy = cv2.GaussianBlur(grad_I_x * grad_I_y, (3, 3), 1.5)
        H = np.array([[S_x_sq, S_xy], [S_xy, S_y_sq]])  # 2x2 x h x w

        det_H = H[0, 0, :, :] * H[1, 1, :, :] - H[0, 1, :, :] * H[1, 0, :, :]
        trace_H = np.trace(H)
        divided_by_zero_map = np.abs(trace_H) < 0.00001
        det_H[divided_by_zero_map] = 0
        trace_H[divided_by_zero_map] = 1

        f_HM = det_H / trace_H
        corner_threshold = CORNER_THRESHOLD
        feat_bitmap = f_HM > corner_threshold
        while np.sum(feat_bitmap) > MAX_FEAT_CANDIDATE_NUM:
            corner_threshold += 500
            feat_bitmap = f_HM > corner_threshold

        feat_locs = np.nonzero(feat_bitmap)
        feat_locs = anms(f_HM, np.array(feat_locs).T)

        descriptors = compute_descriptor_vecs(gray_img, feat_locs)
        descs_feat_locs[i] = (descriptors, feat_locs)

    return descs_feat_locs

C_ROBUST = 0.9
def anms(f_HM, feat_locs, feat_num=250):
    max_r = np.linalg.norm(f_HM.shape)
    candidates = [None] * feat_locs.shape[0]

    feat_strength_i = f_HM[feat_locs[:, 0], feat_locs[:, 1]]
    feat_strength_j = feat_strength_i * C_ROBUST
    fs_i_mat = np.tile(feat_strength_i[:, np.newaxis], (1, feat_strength_i.size))
    fs_j_mat = np.tile(feat_strength_j, (feat_strength_i.size, 1))
    comp_map = fs_i_mat < fs_j_mat
    comp_map_locs = np.nonzero(comp_map)

    d = np.linalg.norm(feat_locs[comp_map_locs[0]] - feat_locs[comp_map_locs[1]], axis=1)
    i = 0
    min_r = max_r
    for ii, dist in zip(comp_map_locs[0], d):
        while i != ii:
            candidates[i] = (feat_locs[i][0], feat_locs[i][1], min_r)
            i += 1
            min_r = max_r
        else:
            if dist < min_r:
                min_r = dist

    for j in range(i, len(candidates)):
        candidates[i] = (feat_locs[i][0], feat_locs[i][1], max_r)

    candidates.sort(key=lambda t: t[2], reverse=True)
    cand_len = len(candidates)

    return np.array(candidates[:feat_num if feat_num < cand_len else cand_len])[:, :-1]

PATCH_SIZE = 4
SAMPLING_GAP = 5
def compute_descriptor_vecs(gray_img, feat_locs):
    # no rotation invarient implementation
    # no wavelet transform
    gray_img = cv2.GaussianBlur(gray_img, (3, 3), 1)
    sampling_y, sampling_x = SAMPLING_GAP * np.mgrid[-PATCH_SIZE: PATCH_SIZE + 1, -PATCH_SIZE: PATCH_SIZE + 1]
    descriptors = [None] * len(feat_locs)
    for i, feat_loc in enumerate(feat_locs):
        patch = ndimage.map_coordinates(gray_img, [feat_loc[0] + sampling_y, feat_loc[1] + sampling_x], order=1)
        descriptors[i] = (patch.ravel() - patch.mean()) / patch.std()

    return np.array(descriptors)

def stitch(imgs, descs_feat_locs):
    pano = imgs[0]
    for i in range(1, len(imgs)):
        dy, dx = compute_displacement(i, descs_feat_locs)
        blend_and_stitch(pano, imgs[i], dy, dx)

    return pano

CLOSE_DISTANCE = 3
def compute_displacement(i, descs_feat_locs):
    kd_tree = spatial.cKDTree(descs_feat_locs[i - 1][0])
    dd, ii = kd_tree.query(descs_feat_locs[i][0], k=1)
    close_points_idxes = dd < CLOSE_DISTANCE
    matched_idxes = ii[close_points_idxes]
    displacements = descs_feat_locs[i - 1][1][matched_idxes] - descs_feat_locs[i][1][close_points_idxes]

    return np.mean(displacements, axis=0)

def blend_and_stitch(pano, img, dy, dx):
    pass