import numpy as np
from scipy.spatial.transform import Rotation as R

def quaternions_to_axes(quaternions):
    # 四元数格式转换 w,x,y,z -> x,y,z,w
    quats_xyzw = np.concatenate([quaternions[:,1:], quaternions[:,0:1]], axis=1)
    rotations = R.from_quat(quats_xyzw)

    # 每个旋转的矩阵 (N, 3, 3)
    matrices = rotations.as_matrix()

    # 提取旋转后的三个轴向量
    x_axes = matrices[:, :, 0]  # 第一列
    y_axes = matrices[:, :, 1]  # 第二列
    z_axes = matrices[:, :, 2]  # 第三列（即法线方向）

    return x_axes, y_axes, z_axes

def points_on_plane(points, normals, eps=0.01):
    base_normal = normals[0]
    base_point = points[0]
    for index, point in enumerate(points):
        if index == 0:
            continue
        vec = point - base_point
        if abs(np.dot(base_normal, vec))>eps:
            return False
        if abs(np.dot(base_normal, normals[index])-1)>eps:
            return False
    return True

def other_points_on_plane(points, normals, eps=0.2):
    base_normal = normals[0]
    base_point = points[0]
    for index, point in enumerate(points):
        if index == 0:
            continue
        vec = point - base_point
        if abs(np.dot(base_normal, vec))>eps:
            return False
        if abs(np.dot(base_normal, normals[index])-1)>eps:
            return False
    return True

def point_on_plane(point, normal, points, normals, eps=0.5):
    if other_points_on_plane(points,normals):
        for p in points:
            vec = normalize(p-point)
            if abs(np.dot(normal,vec))>eps:
                return False
    return True

def points_in_same_sh(sh1, sh2):
    if np.linalg.norm(sh1-sh2)<0.1:
        return True
    else:
        return False
    
def normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("zero vector")
    return v / n