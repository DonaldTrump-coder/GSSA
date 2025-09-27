import torch
from internal.utils.gaussian_model_loader import GaussianModelLoader
from scipy.spatial import cKDTree
from internal.geometry.geometrytools import quaternions_to_axes, points_on_plane, points_in_same_sh, point_on_plane
import numpy as np
from internal.geometry.fill_ellip import fill_in_two_ellips, normalize
from internal.geometry.merge_ellip import merge_two_ellipses_3d
from tqdm import tqdm

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D 支持

if __name__ == "__main__":
    device = torch.device("cuda")
    ckpt_file = "C:\\Users\\10527\\Desktop\\Research of 2DGS\\data\\epoch=28-step=1000.ckpt"
    print(ckpt_file)
    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    model = GaussianModelLoader.initialize_model_from_checkpoint(
            ckpt,
            device=device)
    model.freeze()
    model.pre_activate_all_properties()#用激活函数激活变换模型中的各参数
    means = model.gaussians["means"].detach().cpu().numpy()
    opacities = model.gaussians["opacities"].detach().cpu().numpy()
    scales = model.gaussians["scales"].detach().cpu().numpy()
    rotations = model.gaussians["rotations"].detach().cpu().numpy()
    us, vs, normals = quaternions_to_axes(rotations)
    shs = model.gaussians["shs"].detach().cpu().numpy()[:,0,:]

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
        if np.any(dist[i] > 1.0):
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

    tree = None
    # merge Gaussians
    tree = cKDTree(means)
    merge_index = np.empty(means.shape[0], dtype=object)
    merge_index[:] = None
    dist, idx = tree.query(means, k=6)
    merge_num = 0
    for i in tqdm(range(len(means)), desc="Merging Gaussians"):
        if np.any(dist[i] > 1.0):
            continue
        if not points_on_plane(means[idx[i]], normals[idx[i]]):
            continue
        if np.any(merge_index==i):
            continue
        ellip1 = (means[i], us[i], vs[i], scales[i][0], scales[i][1])
        for n in range(5):
            if np.any(merge_index==idx[i][n+1]):
                continue
            if not points_in_same_sh(shs[i],shs[idx[i][n+1]]):
                continue
            merge_index[i] = idx[i][n+1]
            merge_index[idx[i][n+1]] = i
            ellip2 = (means[idx[i][n+1]], us[idx[i][n+1]], vs[idx[i][n+1]], scales[idx[i][n+1]][0], scales[idx[i][n+1]][1])
            new_gs = merge_two_ellipses_3d(ell1=ellip1, ell2=ellip2, m=128, safety_factor=1.01)

            if new_gs is None:
                continue
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

    mask = np.array([x is not None for x in merge_index])
    merge_index = merge_index[mask].astype(int)
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

def gs_plane(means, opacities, scales, us, vs, normals, shs):
    gaussians_in_plane = []
    tree = cKDTree(means)
    dist, idx = tree.query(means, k=15)
    for i in tqdm(range(len(means)), desc="Judging Planes"):
        if np.any(dist[i] > 1.2):
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
    dist, idx = tree.query(means, k=10)
    for i in tqdm(range(len(means)), desc="Cleaning Gaussians"):
        if dist[i, 2]>1.0:
            gaussians_noise.append(i)
        if points_on_plane(means[idx[i]], normals[idx[i]]):
            continue
        #if dist[i, 1]>0.7 and dist[i, 1]>1.5*np.max(scales[i]):
        if dist[i, 1]>0.8:
            gaussians_noise.append(i)
        #if not point_on_plane(means[i],normals[i],means[idx[i,1:6]], normals[idx[i,1:6]]):
            #gaussians_noise.append(i)
        if opacities[i]<0.15:
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
    dist, idx = tree.query(means, k=21)
    new_means = []
    new_opa = []
    new_scales = []
    new_us = []
    new_vs = []
    new_normals = []
    new_shs = []
    add_num = 0
    for i in tqdm(range(len(means)), desc="Filling Holes"):
        if np.any(dist[i] > 1.2):
            continue
        if not points_on_plane(means[idx[i]], normals[idx[i]]):
            continue
        ellip1 = (means[i], us[i], vs[i], scales[i][0], scales[i][1])
        for n in range(10):
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
    ax.set_box_aspect([1,1,1])  # 坐标轴比例相同
    plt.show()