import numpy as np

def normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("zero vector")
    return v / n

def closest_points_between_lines(P0, u, Q0, v, eps=1e-12):
    """
     P(s)=P0 + s*u, Q(t)=Q0 + t*v 
    :
      P0, u, Q0, v : ndarray shape (3,) 
      eps : 
    :
      P_closest :  (P0 + s*u)
      Q_closest :  (Q0 + t*v)
      s, t      : 
      dist      :  ||P_closest - Q_closest||
      midpoint  : (P_closest + Q_closest) / 2
    """
    P0 = np.asarray(P0, dtype=float)
    u  = np.asarray(u, dtype=float)
    Q0 = np.asarray(Q0, dtype=float)
    v  = np.asarray(v, dtype=float)

    r = P0 - Q0

    A = np.dot(u, u)
    B = np.dot(u, v)
    C = np.dot(v, v)
    D = np.dot(u, r)
    E = np.dot(v, r)

    Delta = A*C - B*B

    if abs(Delta) > eps:
        s = (B*E - C*D) / Delta
        t = (A*E - B*D) / Delta
        P_closest = P0 + s * u
        Q_closest = Q0 + t * v
    else:
        #  (P0 + s u)  Q 
        #  s (P0 + s u - Q0)  u
        #  t = 0 () s  (P0 + s u)  Q0  u 
        #  r r_para = proj_u(r), r_perp = r - r_para ||r_perp||
        #  s = - (u·r) / (u·u) t = 0 
        P_closest = P0
        Q_closest = Q0

    midpoint = 0.5 * (P_closest + Q_closest)
    return midpoint

def fill_in_two_ellips(ell1, ell2):
    C1, ax11, ax12, a1, b1 = ell1 #center coordinate, vector1, vector2, scale1,scale2
    C2, ax21, ax22, a2, b2 = ell2
    vec1 = None
    vec2 = None
    norm1 = None
    norm2 = None
    if np.linalg.norm(C1-C2)<0.001:
        return None

    # judge if has gap exists
    vec = C1-C2
    dist = np.linalg.norm(vec)
    normal = np.cross(ax11, ax12)
    if abs(np.dot(vec, ax11)) >= np.dot(vec, ax12):
        vec1 = ax11
        norm1 = a1
    else:
        vec1 = ax12
        norm1 = b1

    if abs(np.dot(vec, ax21)) >= np.dot(vec, ax22):
        vec2 = ax21
        norm2 = a2
    else:
        vec2 = ax22
        norm2 = b2

    if norm1+norm2 >= 1.5*dist:
        return None
    
    # add ellips
    point = closest_points_between_lines(C1, vec1, C2, vec2, 1e-10) # center of new ellip
    axis1_vec = C1 - point
    scale1 = np.linalg.norm(axis1_vec)*1.1
    if(scale1 < 1e-8 or scale1 > 1.1):
        return None
    axis2_vec = np.cross(normal, axis1_vec)
    scale2 = np.linalg.norm(C2 - point)*1.1
    if(scale2 < 1e-8 or scale2 >1.1):
        return None

    return {
        "center": point,
        "axis_vec1": normalize(axis1_vec),
        "axis_vec2": normalize(axis2_vec),
        "scale1": scale1,
        "scale2": scale2
    }

if __name__ == "__main__":
    P0 = np.array([0.0, 0.0, 0.0])
    u  = np.array([0.0, 1.0, 0.0])
    Q0 = np.array([1.0, 1.0, 0.0])
    v  = np.array([0.0, -1.0, 0.0])
    midpoint = closest_points_between_lines(P0, u, Q0, v)
    print("midpoint =", midpoint)