import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud("/media/allen/新加卷/CityGaussian/outputs/jinguilou_coarse/input.ply")
#mesh = o3d.io.read_triangle_mesh("media/allen/新加卷/CityGaussian/outputs/citygsv2_mc_aerial_sh2_trim/fuse.ply")  # 如果是网格就用这个

# 获取 AABB
aabb = pcd.get_axis_aligned_bounding_box()  # 或 mesh.get_axis_aligned_bounding_box()

# 获取 AABB 的最小和最大点
min_bound = aabb.get_min_bound()
max_bound = aabb.get_max_bound()

# 计算对角线距离（欧氏距离）
diagonal = np.linalg.norm(max_bound - min_bound)

print("AABB 对角线长度为:", diagonal)