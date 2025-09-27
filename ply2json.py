import open3d as o3d
import numpy as np
import json
from scipy.spatial import ConvexHull

# 1. 读取点云
pcd = o3d.io.read_point_cloud("/media/allen/新加卷/CityGaussian/data/geometry_gt/jinguilou/lidar.ply")  # 替换为你的文件路径

# 转为 numpy array
points = np.asarray(pcd.points)

# 选择某一轴（例如 Z 轴）上的 min 和 max 作为 axis_min/max
z_values = points[:, 2]
axis_min = float(np.min(z_values))
axis_max = float(np.max(z_values))

# 获取XY平面上的投影用于构建2D边界（bounding polygon）
xy_points = points[:, :2]  # 取前两列

# 凸包构建2D边界
hull = ConvexHull(xy_points)
polygon_indices = hull.vertices
bounding_polygon = [[float(xy_points[i][0]), float(xy_points[i][1]), 0.0] for i in polygon_indices]

# 构建目标格式
result = {
    "axis_min": round(axis_min, 2),
    "axis_max": round(axis_max, 2),
    "bounding_polygon": bounding_polygon,
    "class_name" : "SelectionPolygonVolume", 
    "orthogonal_axis" : "Z", 
    "version_major" : 1, 
    "version_minor" : 0 
}

# 6. 保存为 JSON 文件
with open("/media/allen/新加卷/CityGaussian/data/geometry_gt/jinguilou/lidar.json", "w") as f:
    json.dump(result, f, indent=2)