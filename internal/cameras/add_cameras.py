import open3d as o3d
from internal.cameras.cameras import Cameras,Camera
import torch
import numpy as np

def visualising_cameras(cameras:list[Camera],pcd_file):
    tensor_list=[]
    mesh_list=[]
    for camera in cameras:
        tensor_list.append(-camera.R.T@camera.T)
        #tensor_list.append(camera.camera_center)

        # depth
        
        depth=10
        bottom_points_cam = torch.tensor([
        [(-camera.cx) * depth / camera.fx, (-camera.cy) * depth / camera.fy, depth],  # 
        [(camera.width - camera.cx) * depth / camera.fx, (-camera.cy) * depth / camera.fy, depth],  # 
        [(camera.width - camera.cx) * depth / camera.fx, (camera.height - camera.cy) * depth / camera.fy, depth],  # 
        [(-camera.cx) * depth / camera.fx, (camera.height - camera.cy) * depth / camera.fy, depth],  # 
        ]).cpu().numpy()
        points_world=(camera.R.cpu().numpy().T)@(bottom_points_cam.T)+np.repeat(-camera.R.cpu().numpy().T@camera.T.cpu().numpy().reshape(3,1),repeats=4,axis=1)
        points_cam=np.vstack([((-camera.R.cpu().numpy().T)@camera.T.cpu().numpy()), points_world.T])

        triangles = np.array([
        [0, 1, 2],  # 
        [0, 2, 3],  # 
        [0, 3, 4],  # 
        [0, 4, 1],  # 
        [1, 4, 3],  # 
        [1, 3, 2]   # 
        ])
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(points_cam)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.paint_uniform_color([240/255,230/255,140/255])
        mesh_list.append(mesh)
        

    result=torch.stack(tensor_list,dim=0)
    result=result.cpu().numpy()
    camera_pcd=o3d.geometry.PointCloud()
    camera_pcd.points = o3d.utility.Vector3dVector(result)


    vis=o3d.visualization.Visualizer()
    vis.create_window(width=800, height=600, window_name="3D Visualization")

    '''
    for point in points_cam_list:
        pcd=o3d.geometry.PointCloud()
        pcd.points=o3d.utility.Vector3dVector(point)
        vis.add_geometry(pcd)
    '''

    #vis.add_geometry(camera_pcd)
    
    for mesh in mesh_list:
        vis.add_geometry(mesh)

    pcd=o3d.io.read_point_cloud(pcd_file)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

def add_cameras(pcd_file,cameras:list[Camera],device):
    pcd=o3d.io.read_point_cloud(pcd_file)#
    samples=0
    center,radius=compute_sphere_center_radius(pcd)#
    center_points=fibonacci_sphere(samples,radius,center)#
    Rs=torch.zeros(samples,3,3,device=device)
    Ts=torch.zeros(samples,3,device=device)

    fx=torch.full((samples,),cameras[0].fx,device=device)
    fy=torch.full((samples,),cameras[0].fy,device=device)
    cx=torch.full((samples,),cameras[0].cx,device=device)
    cy=torch.full((samples,),cameras[0].cy,device=device)
    width=torch.full((samples,),cameras[0].width,device=device)
    height=torch.full((samples,),cameras[0].height,device=device)
    appearance_id=torch.full((samples,),cameras[0].appearance_id,device=device)
    normalized_appearance_id=torch.full((samples,),cameras[0].normalized_appearance_id,device=device)
    camera_type=torch.full((samples,),cameras[0].camera_type,device=device)
    

    for index in range(samples):
        pos=center_points[index]
        pos=pos+(center-pos)*0.4
        z_axis=center-pos
        z_axis = z_axis / np.linalg.norm(z_axis)  # 

        # Z
        temp_up = np.array([0, 1, 0])
        if np.abs(np.dot(z_axis, temp_up)) > 0.99:  # 
            temp_up = np.array([0, 0, 1])
        # X
        x_axis = np.cross(temp_up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)

        # Y
        y_axis = np.cross(x_axis,z_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        R = np.column_stack((x_axis, y_axis,z_axis)).T
        T = -R @ pos

        Rs[index]=torch.tensor(R,device=device)
        Ts[index]=torch.tensor(T,device=device)

        index+=1

    camera=Cameras(
        R=Rs,
        T=Ts,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        width=width,
        height=height,
        appearance_id=appearance_id,
        normalized_appearance_id=normalized_appearance_id,
        distortion_params=None,
        camera_type=camera_type
    )
    for cam in camera:
        cameras.append(cam)

    return cameras

def get_intrinsic(camera:Camera):
    K = np.array([
        [camera.fx.cpu(), 0, camera.cx.cpu()],
        [0, camera.fy.cpu(), camera.cy.cpu()],
        [0, 0, 1]
    ], dtype=np.float64)
    return K

def get_extrinsic(camera:Camera):
    R = np.asarray(camera.R.cpu(), dtype=np.float64).reshape(3, 3)
    T = np.asarray(camera.T.cpu(), dtype=np.float64).reshape(3, 1)
    Rt = np.hstack((R, T))
    return Rt

def compute_sphere_center_radius(pcd):
    points = np.asarray(pcd.points)
    center = np.mean(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    radius = np.max(distances) * 1.2  # 
    return center, radius

def fibonacci_sphere(samples=100, radius=1.0, center=np.zeros(3)):
    """"""
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # 
    
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y1-1
        radius_at_y = np.sqrt(1 - y * y)     # y
        
        theta = phi * i  # 
        
        x = np.cos(theta) * radius_at_y
        z = np.sin(theta) * radius_at_y
        
        points.append(center + np.array([x, y, z]) * radius)
    
    return np.array(points)

if __name__=="__main__":
    add_cameras("/media/allen//CityGaussian/data/geometry_gt/jinguilou_post/lidar.ply")