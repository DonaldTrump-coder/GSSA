from blender.cam_pose_utils.cam_reader import readColmapSceneInfo,read_extrinsics_binary,readColmapCameras
from blender.cam_pose_utils.colmap_loader import qvec2rotmat
from blender.cam_pose_utils.graphic_utils import getWorld2View
from internal.utils import colmap
import open3d as o3d
import numpy as np
from scipy.interpolate import griddata
from PIL import Image

def fill_sparse_depth(depth_map:np.array,width,height,grid_x,grid_y):#最邻近插值
    h, w = height,width

    # 找出有效深度（非inf）
    mask = np.isfinite(depth_map)
    known_points = np.stack((grid_x[mask], grid_y[mask]), axis=-1)
    known_values = depth_map[mask]

    #mask_unknown=~mask
    #unknown_points = np.stack((grid_x[mask_unknown], grid_y[mask_unknown]), axis=-1)

    #for pt in unknown_points:
        #distances = np.linalg.norm(known_points - pt, axis=1)
        #distances[distances == 0] = 1e-6  # 防止除以 0

        #weights = 1 / distances**2
        #value = np.sum(weights * known_values) / np.sum(weights)

        #x, y = pt[0], pt[1]
        #depth_map[y, x] = value
    #return depth_map

    # 所有像素坐标
    all_points = np.stack((grid_x.ravel(), grid_y.ravel()), axis=-1)

    # 使用最近邻插值填充无效区域
    filled = griddata(
        points=known_points,
        values=known_values,
        xi=all_points,
        method='nearest'
    )

    # 重新 reshape 成原图大小
    return filled.reshape((h, w))


#file=open("data/matrix_city/aerial/train/block_all/sparse/0/cameras.bin",'rb')
#values=readColmapSceneInfo("data/matrix_city/aerial/train/block_all")
#values=read_points3D_binary("data/matrix_city/aerial/train/block_all/sparse/0/points3D.bin")
#values=colmap.read_points3D_binary("data/matrix_city/aerial/train/block_all/sparse/0/points3D.bin")
#data = file.read(16 * 4)  # 4个浮点数，每个4字节
#values = struct.unpack('f' * 16, data)  # 解包为4个浮点数
#print(values)

def downsample_ply(path,sampling_ratio=0.3):#点云降采样
    ply=o3d.io.read_point_cloud(path)
    #o3d.visualization.draw_geometries([ply],window_name='1',width=1024,height=768,left=50,top=50,mesh_show_back_face=True)
    downply=ply.random_down_sample(sampling_ratio)
    print(downply)
    o3d.io.write_point_cloud("data/geometry_gt/MC_Aerial/Block_all_ds_downsampled.ply",downply,write_ascii=True)

def save_predicted_depth(depth_map,num):
    num_str=str(num).zfill(4)
    filename=num_str+".png.npy"
    np.save("data/matrix_city/aerial/train/block_all/sampled_depths/"+filename,depth_map)

if __name__=="__main__":
    downsample_ply("data/geometry_gt/MC_Aerial/Block_all_ds.ply",0.06)
    print("ok")

    #ply=o3d.io.read_point_cloud("data/geometry_gt/MC_Aerial/Block_all_ds_downsampled.ply")#加载点云文件
    ply=o3d.io.read_point_cloud("data/geometry_gt/MC_Aerial/Block_all_ds.ply")#加载点云文件
    points = np.asarray(ply.points)

    diameter=np.linalg.norm(np.asarray(ply.get_max_bound())-np.asarray(ply.get_min_bound()))
    radius=diameter*0.2

    pcd=o3d.geometry.PointCloud()
    grid_x, grid_y = np.meshgrid(np.arange(1920), np.arange(1080))

    values=readColmapSceneInfo("data/matrix_city/aerial/train/block_all")#读摄像机内外参
    for camera in values:
        #if(camera.uid<=31):
            #continue
        rotmat=qvec2rotmat(camera.qvec)
        w2c=getWorld2View(rotmat,camera.tvec)#外参矩阵

        intr_array=camera.intr_array
        fx = (intr_array[2] / 2) / np.tan(intr_array[0] / 2)
        fy=fx * intr_array[1]
        cx=intr_array[2]/2
        cy=intr_array[3]/2
        width=intr_array[2]
        height=intr_array[3]
        #camera_intr=o3d.camera.PinholeCameraIntrinsic(width=int(width),height=int(height),fx=fx,fy=fy,cx=cx,cy=cy)
        K=np.array([[fx,  0, cx, 0],
           [ 0, fy, cy, 0],
           [ 0,  0,  1, 0],
           [ 0,  0,  0, 1]])
        
        points_3d= np.hstack((points, np.ones((points.shape[0], 1))))@(w2c.T)
        pcd.points=o3d.utility.Vector3dVector(points_3d[:,0:3])#摄像机坐标系下的点云

        _,pt_map=pcd.hidden_point_removal([0,0,1],radius)
        pcd=pcd.select_by_index(pt_map)
        pcd_points = np.asarray(pcd.points)

        #print(camera.uid)
        #o3d.visualization.draw_geometries([pcd])

        points_3d=np.hstack((pcd_points, np.ones((pcd_points.shape[0], 1))))

        points_2d= points_3d @ (K.T)
        points_2d[:,[0,1]]/=points_2d[:,[2]]
        depth = points_3d[:, 2]  # 深度值

        depth_map = np.full((int(height), int(width)), np.inf)
        w = points_2d[:, 0].astype(int)
        h = points_2d[:, 1].astype(int)
        mask = np.logical_and.reduce((w >= 0, w < width, h >= 0, h < height))
        w, h, depth = w[mask], h[mask], depth[mask]
        #mask=(0 <= w) & (w < width) & (0 <= h) & (h < height)
        #w, h, depth = np.compress(mask, w), np.compress(mask, h), np.compress(mask, depth)

        #for i in range(len(w)):
        #    if depth_map[h[i], w[i]] > depth[i] and depth[i]>=0:
        #        depth_map[h[i], w[i]] = depth[i]

        mask_a = (depth < depth_map[h, w]) & (depth >= 0)
        depth_map[h[mask_a], w[mask_a]] = depth[mask_a]
        
        depth_map=fill_sparse_depth(depth_map,int(width),int(height),grid_x,grid_y)
        depth_map=depth_map[::6,::6]
        #if(camera.uid==25):
            #print(depth_map[120,40],camera.uid)
        #depth_map=1/depth_map#取逆深度
        np.reciprocal(depth_map, out=depth_map)

        save_predicted_depth(depth_map,camera.uid-1)

        #max_depth = np.max(depth_map[np.isfinite(depth_map)])
        #depth_map[~np.isfinite(depth_map)] = max_depth

        #depth_map=o3d.geometry.PointCloud.creat_depth_image_from_point_cloud_with_intrinsics(ply,camera_intr,depth_scale=1000.0,depth_trunc=3.0,stride=1)
        #print(depth_map,camera.uid)