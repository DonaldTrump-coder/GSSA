import torch
from internal.utils.gaussian_model_loader import GaussianModelLoader
from scipy.spatial import cKDTree
from internal.geometry.geometrytools import quaternions_to_axes, points_on_plane, points_in_same_sh, point_on_plane, fit_plane_ransac, pca_planarity
import numpy as np
from internal.geometry.fill_ellip import fill_in_two_ellips, normalize
from internal.geometry.merge_ellip import merge_two_ellipses_3d
from tqdm import tqdm

import matplotlib.pyplot as plt

def gsfilter(means, opacities, scales, us, vs, normals, shs):
    # fill holes
    tree = cKDTree(means)
    dist, idx = tree.query(means, k=6)
    new_means = []
    new_opa = []
    new_scales = []
    new_us = []
    new_vs = []
    new_normals = []
    new_shs = []
    add_num = 0
    for i in tqdm(range(len(means)), desc="Filling Holes"):
        if np.any(dist[i] > 0.5):
            continue
        if not points_on_plane(means[idx[i]], normals[idx[i]]):
            continue
        ellip1 = (means[i], us[i], vs[i], scales[i][0], scales[i][1])
        for n in range(5):
            if not points_in_same_sh(shs[i],shs[idx[i][n+1]]):
                continue
            if idx[i][n+1] == i:
                continue
            ellip2 = (means[idx[i][n+1]], us[idx[i][n+1]], vs[idx[i][n+1]], scales[idx[i][n+1]][0], scales[idx[i][n+1]][1])
            new_gs = fill_in_two_ellips(ell1=ellip1,ell2=ellip2)
            if new_gs is None:
                continue
            new_means.append(new_gs["center"])
            new_opa.append((opacities[i]+opacities[idx[i][n+1]])/2)
            new_scales.append(np.array([new_gs["scale1"], new_gs["scale2"]]))
            new_us.append(new_gs["axis_vec1"])
            new_vs.append(new_gs["axis_vec2"])
            normal = np.cross(new_gs["axis_vec1"],new_gs["axis_vec2"])

            if np.dot(normal, normals[i])<0:
                normal = -normal

            new_normals.append(normalize(normal))
            new_shs.append((shs[i]+shs[idx[i][n+1]])/2)
            add_num+=1

    means = np.vstack([means,np.array(new_means)])
    opacities = np.vstack([opacities,np.array(new_opa)])
    scales = np.vstack([scales,np.array(new_scales)])
    us = np.vstack([us,np.array(new_us)])
    vs = np.vstack([vs,np.array(new_vs)])
    normals = np.vstack([normals,np.array(new_normals)])
    shs = np.vstack([shs,np.array(new_shs)])

    tree = None
    new_means = []
    new_opa = []
    new_scales = []
    new_us = []
    new_vs = []
    new_normals = []
    new_shs = []
    # merge Gaussians
    tree = cKDTree(means)
    merge_index = np.empty(means.shape[0], dtype=object)
    merge_index[:] = None
    dist, idx = tree.query(means, k=7)
    merge_num = 0
    for i in tqdm(range(len(means)), desc="Merging Gaussians"):
        if np.any(dist[i] > 0.5):
            continue
        if not points_on_plane(means[idx[i]], normals[idx[i]]):
            continue
        if merge_index[i] is not None:
            continue
        ellip1 = (means[i], us[i], vs[i], scales[i][0], scales[i][1])
        for n in range(4):
            if merge_index[idx[i][n+1]] is not None:
                continue
            if idx[i][n+1] == i:
                continue
            if not points_in_same_sh(shs[i],shs[idx[i][n+1]]):
                continue
            ellip2 = (means[idx[i][n+1]], us[idx[i][n+1]], vs[idx[i][n+1]], scales[idx[i][n+1]][0], scales[idx[i][n+1]][1])
            new_gs = merge_two_ellipses_3d(ell1=ellip1, ell2=ellip2, m=98, safety_factor=1.01)
            if new_gs is None:
                continue
            merge_index[i] = idx[i][n+1]
            merge_index[idx[i][n+1]] = i
            new_means.append(new_gs["center"])
            new_opa.append((opacities[i]+opacities[idx[i][n+1]])/2)
            new_scales.append(np.array([new_gs["scale1"], new_gs["scale2"]]))
            new_us.append(new_gs["axis_vec1_3"])
            new_vs.append(new_gs["axis_vec2_3"])
            normal = np.cross(new_gs["axis_vec1_3"],new_gs["axis_vec2_3"])

            if np.dot(normal, normals[i])<0:
                normal = -normal

            new_normals.append(normalize(normal))
            new_shs.append((shs[i]+shs[idx[i][n+1]])/2)
            merge_num+=1
            break

    merge_index = np.array([x for x in merge_index if x is not None], dtype=int)
    means = np.delete(means, merge_index, axis=0)
    opacities = np.delete(opacities, merge_index, axis=0)
    scales = np.delete(scales, merge_index, axis=0)
    us = np.delete(us, merge_index, axis=0)
    vs = np.delete(vs, merge_index, axis=0)
    normals = np.delete(normals, merge_index, axis=0)
    shs = np.delete(shs, merge_index, axis=0)

    means = np.vstack([means,np.array(new_means)])
    opacities = np.vstack([opacities,np.array(new_opa)])
    scales = np.vstack([scales,np.array(new_scales)])
    us = np.vstack([us,np.array(new_us)])
    vs = np.vstack([vs,np.array(new_vs)])
    normals = np.vstack([normals,np.array(new_normals)])
    shs = np.vstack([shs,np.array(new_shs)])
    tree = None

    gaussians_in_plane = []
    tree = cKDTree(means)
    dist, idx = tree.query(means, k=7)
    for i in tqdm(range(len(means)), desc="Judging Planes"):
        if not points_on_plane(means[idx[i]], normals[idx[i]]):
            gaussians_in_plane.append(False)
        else:
            gaussians_in_plane.append(True)
    gaussians_in_plane = np.array(gaussians_in_plane)

    return means, opacities, scales, us, vs, normals, shs, gaussians_in_plane

def gs_fill_pca(means, opacities, scales, us, vs, normals, shs,
                planarity_threshold=0.15):
    """
     PCA planarity  points_on_plane  AND
     AND 
    """
    tree = cKDTree(means)
    dist, idx = tree.query(means, k=7)
    new_means, new_opa, new_scales = [], [], []
    new_us, new_vs, new_normals, new_shs = [], [], [], []
    add_num = 0
    for i in tqdm(range(len(means)), desc="Filling Holes [PCA]"):
        if np.any(dist[i] > 0.3):
            continue
        # ---  points_on_planePCA  ---
        planarity, pca_normal = pca_planarity(means[idx[i]])
        if planarity > planarity_threshold:
            continue
        # planarity → 
        w_planar = 1.0 - min(planarity / planarity_threshold, 1.0)
        ellip1 = (means[i], us[i], vs[i], scales[i][0], scales[i][1])
        for n in range(5):
            ellip2 = (means[idx[i][n+1]], us[idx[i][n+1]], vs[idx[i][n+1]],
                       scales[idx[i][n+1]][0], scales[idx[i][n+1]][1])
            new_gs = fill_in_two_ellips(ellip1=ellip1, ellip2=ellip2)
            if new_gs is None:
                continue
            new_means.append(new_gs["center"])
            #  planarity 
            w1 = 0.5 + 0.5 * w_planar
            w2 = 0.5 - 0.5 * w_planar
            new_opa.append(w1 * opacities[i] + w2 * opacities[idx[i][n+1]])
            new_scales.append(np.array([new_gs["scale1"], new_gs["scale2"]]))
            new_us.append(new_gs["axis_vec1"])
            new_vs.append(new_gs["axis_vec2"])
            normal = np.cross(new_gs["axis_vec1"], new_gs["axis_vec2"])
            if np.dot(normal, pca_normal) < 0:
                normal = -normal
            new_normals.append(normalize(normal))
            new_shs.append(w1 * shs[i] + w2 * shs[idx[i][n+1]])
            add_num += 1
    if new_means:
        means = np.vstack([means, np.array(new_means)])
        opacities = np.vstack([opacities, np.array(new_opa)])
        scales = np.vstack([scales, np.array(new_scales)])
        us = np.vstack([us, np.array(new_us)])
        vs = np.vstack([vs, np.array(new_vs)])
        normals = np.vstack([normals, np.array(new_normals)])
        shs = np.vstack([shs, np.array(new_shs)])
    return means, opacities, scales, us, vs, normals, shs

def gs_fill_ransac(means, opacities, scales, us, vs, normals, shs,
                   ransac_dist=0.01, min_inlier_ratio=0.5):
    """
     RANSAC /
    """
    tree = cKDTree(means)
    dist, idx = tree.query(means, k=7)
    new_means, new_opa, new_scales = [], [], []
    new_us, new_vs, new_normals, new_shs = [], [], [], []
    add_num = 0
    for i in tqdm(range(len(means)), desc="Filling Holes [RANSAC]"):
        if np.any(dist[i] > 0.3):
            continue
        # ---  points_on_planeRANSAC  ---
        neighbor_pos = means[idx[i]]
        inlier_mask, inlier_ratio = fit_plane_ransac(neighbor_pos,
                                                      dist_thresh=ransac_dist)
        if inlier_ratio < min_inlier_ratio:
            continue
        inlier_kk = np.where(inlier_mask)[0]
        inlier_kk = inlier_kk[inlier_kk != 0]  # 
        if len(inlier_kk) == 0:
            continue
        ellip1 = (means[i], us[i], vs[i], scales[i][0], scales[i][1])
        for kk in inlier_kk:
            ellip2 = (means[idx[i][kk]], us[idx[i][kk]], vs[idx[i][kk]],
                       scales[idx[i][kk]][0], scales[idx[i][kk]][1])
            new_gs = fill_in_two_ellips(ell1=ellip1, ell2=ellip2)
            if new_gs is None:
                continue
            new_means.append(new_gs["center"])
            #  = 
            w = inlier_ratio
            new_opa.append(w * opacities[i] + (1 - w) * opacities[idx[i][kk]])
            new_scales.append(np.array([new_gs["scale1"], new_gs["scale2"]]))
            new_us.append(new_gs["axis_vec1"])
            new_vs.append(new_gs["axis_vec2"])
            normal = np.cross(new_gs["axis_vec1"], new_gs["axis_vec2"])
            if np.dot(normal, normals[i]) < 0:
                normal = -normal
            new_normals.append(normalize(normal))
            new_shs.append(w * shs[i] + (1 - w) * shs[idx[i][kk]])
            add_num += 1
    if new_means:
        means = np.vstack([means, np.array(new_means)])
        opacities = np.vstack([opacities, np.array(new_opa)])
        scales = np.vstack([scales, np.array(new_scales)])
        us = np.vstack([us, np.array(new_us)])
        vs = np.vstack([vs, np.array(new_vs)])
        normals = np.vstack([normals, np.array(new_normals)])
        shs = np.vstack([shs, np.array(new_shs)])
    return means, opacities, scales, us, vs, normals, shs

def rot_filter(means, opacities, scales, rotations, shs):
    norms = np.linalg.norm(rotations, axis=-1)
    mask = norms > 1e-16
    return means[mask], opacities[mask], scales[mask], rotations[mask], shs[mask]

def opa_norm(opacities):
    vmin = opacities.min()
    vmax = opacities.max()
    opacities = (opacities-vmin)/(vmax-vmin)

    hist, bins = np.histogram(opacities, bins=2048, range = (0,1), density=True)
    cdf = hist.cumsum()
    cdf = cdf / cdf[-1]
    opacities = np.interp(opacities, bins[:-1], cdf)
    return opacities

def gs_plane(means, opacities, scales, us, vs, normals, shs):
    gaussians_in_plane = []
    tree = cKDTree(means)
    dist, idx = tree.query(means, k=15)
    for i in tqdm(range(len(means)), desc="Judging Planes"):
        if np.any(dist[i] > 0.8):
            gaussians_in_plane.append(False)
        elif not points_on_plane(means[idx[i]], normals[idx[i]]):
            gaussians_in_plane.append(False)
        else:
            gaussians_in_plane.append(True)
    gaussians_in_plane = np.array(gaussians_in_plane)
    return gaussians_in_plane

def gs_clean(means, opacities, scales, us, vs, normals, shs):
    gaussians_noise = []
    tree = cKDTree(means)
    dist, idx = tree.query(means, k=5)
    for i in tqdm(range(len(means)), desc="Cleaning Gaussians"):
        if dist[i, 2]>1.0 * 0.5:
            gaussians_noise.append(i)
        if points_on_plane(means[idx[i]], normals[idx[i]]):
            continue
        #if dist[i, 1]>0.7 and dist[i, 1]>1.5*np.max(scales[i]):
        if dist[i, 1]>0.8 * 0.5:
            gaussians_noise.append(i)
        #if not point_on_plane(means[i],normals[i],means[idx[i,1:6]], normals[idx[i,1:6]]):
            #gaussians_noise.append(i)
        if opacities[i]<0.00:
            gaussians_noise.append(i)
    gaussians_noise=np.array(gaussians_noise)
    means = np.delete(means, gaussians_noise, axis=0)
    opacities = np.delete(opacities, gaussians_noise, axis=0)
    scales = np.delete(scales, gaussians_noise, axis=0)
    us = np.delete(us, gaussians_noise, axis=0)
    vs = np.delete(vs, gaussians_noise, axis=0)
    normals = np.delete(normals, gaussians_noise, axis=0)
    shs = np.delete(shs, gaussians_noise, axis=0)
    return means, opacities, scales, us, vs, normals, shs

def gs_fill(means, opacities, scales, us, vs, normals, shs):
    # fill holes
    tree = cKDTree(means)
    dist, idx = tree.query(means, k=7)
    new_means = []
    new_opa = []
    new_scales = []
    new_us = []
    new_vs = []
    new_normals = []
    new_shs = []
    add_num = 0
    for i in tqdm(range(len(means)), desc="Filling Holes"):
        if np.any(dist[i] > 0.3):
            continue
        if not points_on_plane(means[idx[i]], normals[idx[i]]):
            continue
        ellip1 = (means[i], us[i], vs[i], scales[i][0], scales[i][1])
        for n in range(5):
            ellip2 = (means[idx[i][n+1]], us[idx[i][n+1]], vs[idx[i][n+1]], scales[idx[i][n+1]][0], scales[idx[i][n+1]][1])
            new_gs = fill_in_two_ellips(ell1=ellip1,ell2=ellip2)
            if new_gs is None:
                continue
            new_means.append(new_gs["center"])
            new_opa.append((opacities[i]+opacities[idx[i][n+1]])/2)
            new_scales.append(np.array([new_gs["scale1"], new_gs["scale2"]]))
            new_us.append(new_gs["axis_vec1"])
            new_vs.append(new_gs["axis_vec2"])
            normal = np.cross(new_gs["axis_vec1"],new_gs["axis_vec2"])

            if np.dot(normal, normals[i])<0:
                normal = -normal

            new_normals.append(normalize(normal))
            new_shs.append((shs[i]+shs[idx[i][n+1]])/2)
            add_num+=1

    means = np.vstack([means,np.array(new_means)])
    opacities = np.vstack([opacities,np.array(new_opa)])
    scales = np.vstack([scales,np.array(new_scales)])
    us = np.vstack([us,np.array(new_us)])
    vs = np.vstack([vs,np.array(new_vs)])
    normals = np.vstack([normals,np.array(new_normals)])
    shs = np.vstack([shs,np.array(new_shs)])
    return means, opacities, scales, us, vs, normals, shs

def plot_points_3d(pts, color='b', size=20, label=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=color, s=size, label=label)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if label:
        ax.legend()
    ax.set_box_aspect([1,1,1])  # 
    plt.show()