import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D 支持

def normalize(x, eps=1e-12):
    n = np.linalg.norm(x)
    if n < eps:
        return x
    return x / n

def ellipse_Sigma2_from_axis_dirs_3d(a, b, axis1_3, axis2_3, e1, e2):
    """
    返回平面基 (e1,e2) 下的 2x2 Sigma
    """
    v1 = np.array([np.dot(axis1_3, e1), np.dot(axis1_3, e2)])
    v2 = np.array([np.dot(axis2_3, e1), np.dot(axis2_3, e2)])
    Sigma2 = (a*a) * np.outer(v1, v1) + (b*b) * np.outer(v2, v2)
    return Sigma2

def support_point_2d(c2, Sigma2, u2):
    denom = np.sqrt(float(u2.T @ Sigma2 @ u2))
    x2 = c2 + (Sigma2 @ u2) / denom
    h = float(u2.T @ x2)
    return x2, h

def mvee(points, tol=1e-7, max_iter=1000):
    P = np.asarray(points)
    N, d = P.shape
    Q = np.hstack((P, np.ones((N,1))))
    u = np.ones(N) / N
    for _ in range(max_iter):
        X = (Q.T * u) @ Q
        try:
            M = np.diag(Q @ np.linalg.inv(X) @ Q.T)
        except np.linalg.LinAlgError:
            X += 1e-12 * np.eye(d+1)
            M = np.diag(Q @ np.linalg.inv(X) @ Q.T)
        j = np.argmax(M)
        max_M = M[j]
        step = (max_M - d - 1) / ((d+1) * (max_M - 1.0))
        new_u = (1 - step) * u
        new_u[j] += step
        if np.linalg.norm(new_u - u) < tol:
            u = new_u
            break
        u = new_u
    c = (P.T @ u).reshape(d,)
    P_centered = P - c
    S = (P_centered.T * u) @ P_centered
    A = np.linalg.inv(S) / d
    return c, A

def ellipse_params_from_A_2d(c2, A):
    Sigma = np.linalg.inv(A)
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    a = np.sqrt(eigvals[1])
    b = np.sqrt(eigvals[0])
    v = eigvecs[:,1]
    theta = np.arctan2(v[1], v[0])
    return c2, a, b, theta, Sigma

def merge_two_ellipses_3d(ell1, ell2, m=128, safety_factor=1.0):
    """
    合并两个 3D 椭圆 (同一平面内).
    ell = (C3, axis1_unit_3, axis2_unit_3, a, b)
    返回: {"center", "axis_dir1_3", "axis_dir2_3", "scale1", "scale2"}
    """
    C1, ax11, ax12, a1, b1 = ell1
    C2, ax21, ax22, a2, b2 = ell2
    if np.linalg.norm(C1-C2)<0.001:
        return None

    # 平面法向
    n1 = np.cross(ax11, ax12)
    n2 = np.cross(ax21, ax22)
    n = n1 + n2
    if np.linalg.norm(n) < 1e-12:
        return None
    n = normalize(n)

    # 选择平面原点 = (C1+C2)/2 在平面上的投影
    mid = 0.5*(C1 + C2)

    # 平面基 (e1,e2)
    e1 = ax11 - np.dot(ax11, n)*n
    if np.linalg.norm(e1) < 1e-12:
        v = C2 - C1
        e1 = v - np.dot(v, n)*n
    e1 = normalize(e1)
    e2 = normalize(np.cross(n, e1))

    def to_plane_coords(C):
        vec = C - mid
        return np.array([np.dot(vec, e1), np.dot(vec, e2)])

    c1_2 = to_plane_coords(C1)
    c2_2 = to_plane_coords(C2)

    Sigma1 = ellipse_Sigma2_from_axis_dirs_3d(a1, b1, ax11, ax12, e1, e2)
    Sigma2 = ellipse_Sigma2_from_axis_dirs_3d(a2, b2, ax21, ax22, e1, e2)

    # 生成支撑点 (取更远的那个点)
    thetas = np.linspace(0, 2*np.pi, m, endpoint=False)
    pts2 = []
    for th in thetas:
        u = np.array([np.cos(th), np.sin(th)])
        x1, h1 = support_point_2d(c1_2, Sigma1, u)
        x2, h2 = support_point_2d(c2_2, Sigma2, u)
        pts2.append(x1 if h1 >= h2 else x2)
    pts2 = np.array(pts2)

    # MVEE
    c_mve2, A_mve = mvee(pts2, tol=1e-8, max_iter=2000)
    if np.linalg.cond(A_mve) > 1e12:
        A_mve += 1e-12 * np.eye(2)

    c2d, a_out, b_out, theta_out, _ = ellipse_params_from_A_2d(c_mve2, A_mve)
    a_out *= safety_factor
    b_out *= safety_factor

    center3 = mid + e1*c2d[0] + e2*c2d[1]
    dir1_2 = np.array([np.cos(theta_out), np.sin(theta_out)])
    dir2_2 = np.array([-np.sin(theta_out), np.cos(theta_out)])

    axis_dir1_3 = normalize(dir1_2[0]*e1 + dir1_2[1]*e2)
    axis_dir2_3 = normalize(dir2_2[0]*e1 + dir2_2[1]*e2)

    return {
        "center": center3,
        "axis_vec1_3": axis_dir1_3,  # 单位方向向量
        "axis_vec2_3": axis_dir2_3,
        "scale1": a_out,             # 半长轴
        "scale2": b_out              # 半短轴
    }

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


if __name__ == "__main__":
    ell1 = (np.array([0.0,0.0,0.0]), normalize(np.array([1,0,0])), normalize(np.array([0,1,0])), 3.0, 1.0)
    ell2 = (np.array([4.0,0.2,0.0]), normalize(np.array([0.98,0.1,0])), normalize(np.array([-0.1,0.98,0])), 2.0, 0.8)

    res = merge_two_ellipses_simple_3d(ell1, ell2, m=128, safety_factor=1.01)
    ok, max_v = verify_coverage_simple(ell1, ell2, res)
    print("covered?", ok, "max value:", max_v)