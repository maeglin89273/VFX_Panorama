import numpy as np
import cv2
import scipy
from scipy import ndimage, spatial
import utils

def stitch_panorama(frag_imgs, focal_len):
    cylinder_imgs, valid_y = cylinder_warp(frag_imgs, focal_len)
    descs_feat_locs = msop(cylinder_imgs, valid_y)
    return stitch(cylinder_imgs, descs_feat_locs, valid_y)

def cylinder_warp(frag_imgs, focal_len):
    cylinder_imgs = [None] * len(frag_imgs)

    #assume every fragment image has the same dimensions
    cylinder_h = frag_imgs[0].shape[0]
    cylinder_w = int(focal_len * 2 * np.arctan(frag_imgs[0].shape[1] / (focal_len * 2)))
    half_img_h = frag_imgs[0].shape[0] / 2
    half_img_w = frag_imgs[0].shape[1] / 2
    half_cy_h = cylinder_h / 2
    half_cy_w = cylinder_w / 2
    focal_len_sq = focal_len ** 2
    valid_y = int(half_cy_h - (focal_len * half_cy_h / np.sqrt(focal_len_sq + half_img_w ** 2)))

    for i, frag_img in enumerate(frag_imgs):
        print('cylinder warp on image %s' % i)
        cylinder_proj = np.zeros((cylinder_h, cylinder_w, frag_img.shape[2]), dtype=frag_img.dtype)
        for y in range(cylinder_proj.shape[0]):
            for x in range(cylinder_proj.shape[1]):
                theta = (x - half_cy_w) / focal_len
                c_img_x = focal_len * np.tan(theta)
                img_x = c_img_x + half_img_w
                img_y = half_img_h - (half_cy_h - y) * np.sqrt(focal_len_sq + c_img_x ** 2) / focal_len
                if 0 <= img_y < frag_img.shape[0] and 0 <= img_x < frag_img.shape[1]:
                    cylinder_proj[y, x, :] = frag_img[int(img_y), int(img_x), :]

        cylinder_imgs[i] = cylinder_proj

    # since the cylinder projection has some invalid area(i.e no projected values),
    #  valid_y indicates the y coordinate that projection starts
    return cylinder_imgs, valid_y



CORNER_THRESHOLD = 5000
MAX_FEAT_CANDIDATE_NUM = 3000
MARGIN = 2
def msop(imgs, valid_y):
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
        utils.show_heatmap(f_HM, gray_img)

        corner_threshold = CORNER_THRESHOLD
        #We are not going use too much candidate feature point (around MAX_FEAT_CANDIDATE_NUM only)
        while True:
            feat_mask = f_HM > corner_threshold
            feat_mask[:valid_y + MARGIN] = False
            feat_mask[-(valid_y + MARGIN):] = False

            if np.sum(feat_mask) > MAX_FEAT_CANDIDATE_NUM:
                corner_threshold += 500
            else:
                break

        feat_locs = np.nonzero(feat_mask)
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

    return np.array(candidates[:feat_num if feat_num < cand_len else cand_len], dtype=int)[:, :-1]

PATCH_SIZE = 4
SAMPLING_GAP = 5
def compute_descriptor_vecs(gray_img, feat_locs):
    # no rotation invarient, since it's not suit for the panorama during feature matching
    # no wavelet transform
    h, w = gray_img.shape
    gray_img = cv2.GaussianBlur(gray_img, (3, 3), 1)

    sampling_y, sampling_x = SAMPLING_GAP * np.mgrid[-PATCH_SIZE: PATCH_SIZE + 1, -PATCH_SIZE: PATCH_SIZE + 1]
    descriptors = [None] * len(feat_locs)
    for i, feat_loc in enumerate(feat_locs):
        patch = ndimage.map_coordinates(gray_img, [feat_loc[0] + sampling_y, feat_loc[1] + sampling_x], order=1)
        descriptors[i] = (patch.ravel() - patch.mean()) / patch.std()

    return np.array(descriptors)


def stitch(imgs, descs_feat_locs, valid_y):
    pano = imgs[0]
    total_dy = 0.0
    for i in range(1, len(imgs)):
        dy, dx = compute_displacement(imgs, i, descs_feat_locs)
        pano = blend_and_stitch(pano, imgs[i], dy, dx)
        total_dy += dy

    #assume the panorama is taken with tripods, so it only leans gradually
    # I use a shear matrix to adjust it
    total_dx = pano.shape[1]
    shear_mat = np.array([[1, 0, 0], [total_dy / total_dx, 1, 0]])

    pano = np.clip(pano, 0, 255)
    pano = pano.astype(imgs[0].dtype)
    pano = cv2.warpAffine(pano, shear_mat, (pano.shape[1], pano.shape[0]))
    if total_dy < 0:
        pano = pano[:int(total_dy)]
    elif total_dy > 0:
        pano = pano[int(total_dy):]

    #remove the invalid area that cylinder projection not uses
    return pano[valid_y: -valid_y]

CLOSE_DISTANCE = 1.0
MIN_MATCHED_FEAT_NUM = 15
def compute_displacement(imgs, i, descs_feat_locs):
    # parameter imgs is for plotting features

    kd_tree = spatial.cKDTree(descs_feat_locs[i - 1][0])
    dd, ii = kd_tree.query(descs_feat_locs[i][0], k=1)

    close_distance = CLOSE_DISTANCE
    close_points_filter = dd < close_distance
    while np.sum(close_points_filter) < MIN_MATCHED_FEAT_NUM:
        close_distance += 0.5
        close_points_filter = dd < close_distance

    # print('matched feature number %s' % np.sum(close_points_filter))
    matched_idxes = ii[close_points_filter]
    pre_img_candidate_feat_points = descs_feat_locs[i - 1][1][matched_idxes]
    img_candidate_feat_points = descs_feat_locs[i][1][close_points_filter]
    displacements = pre_img_candidate_feat_points - img_candidate_feat_points
    displacements, inlier_mask = remove_outlier(displacements)

    utils.show_imgs_with_feats((imgs[i], imgs[i - 1]), (img_candidate_feat_points[inlier_mask], pre_img_candidate_feat_points[inlier_mask]))

    # print('outlier removed matched feature number %s' % displacements.shape[0])
    return np.mean(displacements, axis=0).astype(int)

def remove_outlier(vecs):
    vecs_mean = np.mean(vecs, axis=0)
    vecs_std = np.std(vecs, axis=0)
    inlier_mask = np.all(vecs - vecs_mean < vecs_std, axis=1)
    return vecs[inlier_mask], inlier_mask


def blend_and_stitch(pano, img, dy, dx):
    img_cpy = img.astype(float)
    o_pano_shape = pano.shape
    new_pano_shape = (o_pano_shape[0] + np.abs(dy), o_pano_shape[1] + np.abs(dx), o_pano_shape[2])
    new_pano = np.zeros(new_pano_shape)

    overlap_x = img_cpy.shape[1] - np.abs(dx)
    h_to_l_weights = np.linspace(1, 0, overlap_x)
    l_to_h_weights = np.linspace(0, 1, overlap_x)
    o_pano_begin_y = 0
    o_pano_end_y = o_pano_shape[0]
    img_begin_y = dy
    img_end_y = img_cpy.shape[0] + dy

    if dy < 0:
        o_pano_begin_y += -dy
        o_pano_end_y += -dy
        img_begin_y = 0
        img_end_y = img.shape[0]

    if dx < 0:
        o_pano_begin_x = -dx
        o_pano_end_x = o_pano_shape[1] - dx
        new_pano[o_pano_begin_y: o_pano_end_y, o_pano_begin_x: o_pano_end_x] = pano
        blend_start = o_pano_begin_x
        blend_end = overlap_x + o_pano_begin_x
        new_pano[:, blend_start: blend_end] *= (np.tile(l_to_h_weights, (new_pano.shape[0], 1))[:, :, np.newaxis])
        img_cpy[:, o_pano_begin_x:] *= (np.tile(h_to_l_weights, (img_cpy.shape[0], 1))[:, :, np.newaxis])
        new_pano[img_begin_y: img_end_y, :img_cpy.shape[1]] += img_cpy
    else:
        o_pano_begin_x = 0
        o_pano_end_x = o_pano_shape[1]
        new_pano[o_pano_begin_y: o_pano_end_y, o_pano_begin_x: o_pano_end_x] = pano
        blend_start = o_pano_shape[1] - overlap_x
        blend_end = o_pano_shape[1]
        new_pano[:, blend_start: blend_end] *= (np.tile(h_to_l_weights, (new_pano.shape[0], 1))[:, :, np.newaxis])
        img_cpy[:, :overlap_x] *= (np.tile(l_to_h_weights, (img_cpy.shape[0], 1))[:, :, np.newaxis])
        new_pano[img_begin_y: img_end_y, blend_start:] += img_cpy

    return new_pano