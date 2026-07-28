import numpy as np
from scipy.spatial.transform import Rotation as R

def quaternions_to_axes(quaternions):
    #  w,x,y,z -> x,y,z,w
    quats_xyzw = np.concatenate([quaternions[:,1:], quaternions[:,0:1]], axis=1)
    rotations = R.from_quat(quats_xyzw)

    #  (N, 3, 3)
    matrices = rotations.as_matrix()

    x_axes = matrices[:, :, 0]  # 
    y_axes = matrices[:, :, 1]  # 
    z_axes = matrices[:, :, 2]  # 

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

def pca_planarity(points):
    """
    PCA  k 
     (planarity, )
    planarity = λ₃ / (λ₁+λ₂+λ₃)
    """
    centered = points - points.mean(axis=0)
    cov = centered.T @ centered / (len(points) - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigh : l1 ≤ l2 ≤ l3
    l1, l2, l3 = eigvals[0], eigvals[1], eigvals[2]
    planarity = l1 / (l1 + l2 + l3) if (l1 + l2 + l3) > 1e-10 else 1.0
    normal = eigvecs[:, 0]  #  = 
    return planarity, normal

def fit_plane_ransac(points, n_iter=100, dist_thresh=0.01):
    """
    RANSAC 
     (mask, )
    """
    best_inliers = None
    best_n = 0
    N = len(points)
    for _ in range(n_iter):
        idx = np.random.choice(N, 3, replace=False)
        p0, p1, p2 = points[idx[0]], points[idx[1]], points[idx[2]]
        normal = np.cross(p1 - p0, p2 - p0)
        if np.linalg.norm(normal) < 1e-10:
            continue
        normal = normalize(normal)
        distances = np.abs((points - p0) @ normal)
        inliers = distances < dist_thresh
        n = inliers.sum()
        if n > best_n:
            best_n = n
            best_inliers = inliers
    return best_inliers, best_n / N

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