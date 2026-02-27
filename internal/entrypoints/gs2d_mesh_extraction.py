from dataclasses import dataclass
import os
import torch
import open3d as o3d
from jsonargparse import CLI
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.utils.gs2d_meshing_utils import GS2DMeshingUtils, post_process_mesh
from internal.cameras.add_cameras import visualising_cameras,add_cameras,get_extrinsic,get_intrinsic
from internal.renderers import meshing_2dgs_renderer
import TSDF_forGS
from internal.geometry.texture_mapping import mapping
import numpy as np
from tqdm import tqdm
from internal.models.vanilla_gaussian import quaternions_to_axes
from internal.geometry.gsfilter import gsfilter, gs_fill, gs_plane, gs_clean, opa_norm, rot_filter
from internal.geometry.filter import filter_multiple
from internal.utils.sh_utils import SH2RGB
from scipy.spatial import Delaunay
import time

def extract_bounding(ply_path:str):
    ply=o3d.io.read_point_cloud(ply_path)
    xmin,ymin,zmin=ply.get_min_bound()
    xmax,ymax,zmax=ply.get_max_bound()
    return np.array([[xmin,ymin,zmin,xmax,ymax,zmax]],dtype=np.float64)

def extract_bounding_from_np(means):
    xmin,ymin,zmin=means.min(axis=0)
    xmax,ymax,zmax=means.max(axis=0)
    return np.array([[xmin,ymin,zmin,xmax,ymax,zmax]],dtype=np.float64)

def densify_point_cloud(pcd, radius=1, n_samples=3):
    pts = np.asarray(pcd.points)
    tree = o3d.geometry.KDTreeFlann(pcd)
    new_pts = []
    for p in pts:
        [_, idx, _] = tree.search_radius_vector_3d(p, radius)
        if len(idx) >= 3:
            local = pts[idx]
            mean = local.mean(axis=0)
            cov = np.cov(local.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            normal = eigvecs[:, np.argmin(eigvals)]
            tangent = np.cross(normal, np.array([1, 0, 0]))
            if np.linalg.norm(tangent) < 1e-3:
                tangent = np.cross(normal, np.array([0, 1, 0]))
            tangent /= np.linalg.norm(tangent)
            bitangent = np.cross(normal, tangent)
            for _ in range(n_samples):
                a, b = np.random.randn(2) * (radius * 0.3)
                new_pts.append(p + a * tangent + b * bitangent)
    all_pts = np.vstack([pts, new_pts])
    pcd_out = o3d.geometry.PointCloud()
    pcd_out.points = o3d.utility.Vector3dVector(all_pts)
    return pcd_out

def Poisson_reconstruction(means: np.ndarray, normals: np.ndarray, shs: np.ndarray = None):
    colors = SH2RGB(shs)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(means)
    point_cloud.normals = o3d.utility.Vector3dVector(normals)

    if colors is not None:
        point_cloud.colors = o3d.utility.Vector3dVector(colors)

    print("正在Poission重建")
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=9)

    # 如果有颜色，用 KDTree 将颜色映射到网格顶点
    if colors is not None:
        print("使用 KDTree 映射颜色到网格顶点...")
        # 构建 KDTree
        pcd_tree = o3d.geometry.KDTreeFlann(point_cloud)
        mesh_colors = []
        mesh_vertices = np.asarray(mesh.vertices)

        for v in mesh_vertices:
            _, idx, _ = pcd_tree.search_knn_vector_3d(v, 1)  # 最近邻
            mesh_colors.append(colors[idx[0]])

        mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(mesh_colors))

    # 保存网格
    o3d.io.write_triangle_mesh("poisson_reconstructed.ply", mesh)

def Den(means: np.ndarray, shs: np.ndarray = None):
    colors = SH2RGB(shs)
    # 3D Delaunay 剖分
    print("正在Denaulay重建")
    tri = Delaunay(means)

    faces = []
    for tet in tri.simplices:
        # 每个四面体 4 个三角面
        for i in range(4):
            face = np.delete(tet, i)
            faces.append(face)

    faces = np.array(faces)

    # === 去重（因为内部面会重复两次，顶点顺序不同）===
    faces_sorted = np.sort(faces, axis=1)
    faces_unique = np.unique(faces_sorted, axis=0)

    # 构建 Open3D 网格
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(means)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    # 添加颜色
    if colors is not None:
        # 顶点颜色一一对应
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    # 保存
    o3d.io.write_triangle_mesh("delaunay_surface_mesh.ply", mesh)

def alpha_shape(means: np.ndarray, shs: np.ndarray = None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(means)
    colors = SH2RGB(shs)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    #pcd_dense = densify_point_cloud(pcd, radius=1, n_samples=5)
    
    # === Alpha Shape 重建 ===
    print(f"正在进行 Alpha Shape 重建 (alpha={5}) ...")
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, 0.03)
    mesh.compute_vertex_normals()

    if colors is not None:
        print("正在将点云颜色映射到网格 ...")
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        mesh_colors = []
        for v in np.asarray(mesh.vertices):
            [_, idx, _] = pcd_tree.search_knn_vector_3d(v, 1)  # 找最近的1个点
            mesh_colors.append(colors[idx[0]])
        mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)
    
    # === 保存 ===
    o3d.io.write_triangle_mesh("alpha_surface_mesh.ply", mesh)


#存储数据的类
@dataclass
class CLIArgs:
    model_path: str

    dataset_path: str = None

    voxel_size: float = -1.

    depth_trunc: float = -1.

    sdf_trunc: float = -1.

    num_cluster: int = 50

    unbounded: bool = False

    mesh_res: int = 1024


def main():
    args = CLI(CLIArgs)

    device = torch.device("cuda")

    # load ckpt
    loadable_file = GaussianModelLoader.search_load_file(args.model_path)
    #提取ckpt文件路径
    #loadable_file = "/media/allen/新加卷/CityGaussian/outputs/jinguilou_post/checkpoints/epoch=811-step=30000.ckpt"
    #loadable_file = "/media/allen/新加卷/CityGaussian/outputs/jinguilou_post/checkpoints/epoch=28-step=1000.ckpt"

    print(loadable_file)
    start = time.time()
    dataparser_config = None
    if loadable_file.endswith(".ckpt"):#对于ckpt模型文件
        ckpt = torch.load(loadable_file, map_location="cpu")#把模型加载到GPU中
        # initialize model
        model = GaussianModelLoader.initialize_model_from_checkpoint(
            ckpt,
            device=device,
        )
        #从ckpt文件中读取并加载模型，存在cpu中

        model.freeze()
        model.pre_activate_all_properties()#用激活函数激活变换模型中的各参数

        ckpt["hyper_parameters"]["renderer"]=meshing_2dgs_renderer.Meshing2DGSRenderer()
        # initialize renderer
        renderer = GaussianModelLoader.initialize_renderer_from_checkpoint(
            ckpt,
            stage="validate",
            device=device,
        )
        try:
            dataparser_config = ckpt["datamodule_hyper_parameters"]["parser"]
        except:
            pass

        dataset_path = ckpt["datamodule_hyper_parameters"]["path"]
        if args.dataset_path is not None:
            dataset_path = args.dataset_path
    else:
        dataset_path = args.dataset_path
        if dataset_path is None:
            cfg_args_file = os.path.join(args.model_path, "cfg_args")
            try:
                from argparse import Namespace
                with open(cfg_args_file, "r") as f:
                    cfg_args = eval(f.read())
                dataset_path = cfg_args.source_path
            except Exception as e:
                print("Can not parse `cfg_args`: {}".format(e))
                print("Please specific the data path via: `--dataset_path`")
                exit(1)

        model, renderer = GaussianModelLoader.initialize_model_and_renderer_from_ply_file(
            loadable_file,
            device=device,
            eval_mode=True,
            pre_activate=True,
        )
    if dataparser_config is None:
        from internal.dataparsers.colmap_dataparser import Colmap
        dataparser_config = Colmap()

    
    """
    tsdfstart = time.time()
    
    dataparser_outputs = dataparser_config.instantiate(#用数据路径等参数初始化
        path=dataset_path,
        output_path=os.getcwd(),
        global_rank=0,
    ).get_outputs()#处理好相片、深度对齐的数据
    cameras = [i.to_device(device) for i in dataparser_outputs.train_set.cameras]

    # set the active_sh to 0 to export only diffuse texture
    model.active_sh_degree = 0#只保留基础球谐函数
    bg_color = torch.zeros((3,), dtype=torch.float, device=device)#背景颜色
    maps = GS2DMeshingUtils.render_views(model, renderer, cameras, bg_color)#再一次渲染得到的深度图、rgb图
    bound = GS2DMeshingUtils.estimate_bounding_sphere(cameras)

    name = 'fuse.ply'
    _, radius = bound
    depth_trunc = (radius * 2.0) if args.depth_trunc < 0 else args.depth_trunc
    voxel_size = (depth_trunc / args.mesh_res) if args.voxel_size < 0 else args.voxel_size
    sdf_trunc = 5.0 * voxel_size if args.sdf_trunc < 0 else args.sdf_trunc
    mesh = GS2DMeshingUtils.extract_mesh_bounded(maps=maps, cameras=cameras, voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)
    output_dir = args.model_path
    if os.path.isfile(output_dir):
        output_dir = os.path.dirname(output_dir)
    o3d.io.write_triangle_mesh(os.path.join(output_dir, name), mesh)
    end = time.time()
    tsdfend = time.time()
    tsdfdt = tsdfend-tsdfstart
    print(f"TSDF time: {end-start}sec")
    print("mesh saved at {}".format(os.path.join(output_dir, name)))
    return
    """
    """
    means=model.gaussians["means"].cpu().numpy()
    scales=model.gaussians["scales"].cpu().numpy()
    rotations=model.gaussians["rotations"].cpu().numpy()
    opacities=model.gaussians["opacities"].cpu().numpy()
    shs = model.gaussians["shs"].detach().cpu().numpy()[:,0,:]
    means, opacities, scales, rotations, shs = rot_filter(means,opacities,scales, rotations, shs)
    us,vs,normals=quaternions_to_axes(rotations)
    bounding=extract_bounding("data/geometry_gt/jinguilou_post/lidar.ply")
    dx=(bounding[0][0]+bounding[0][3])/2-bounding[0][0]
    dy=(bounding[0][1]+bounding[0][4])/2-bounding[0][1]
    dz=(bounding[0][2]+bounding[0][5])/2-bounding[0][2]
    bounding[0][0]=-2.099851-4
    bounding[0][3]=-2.099851+4
    bounding[0][1]=1.987608-4
    bounding[0][4]=1.987608+4
    bounding[0][2]=4.158119-4
    bounding[0][5]=4.158119+4
    mask = (
        (means[:, 0] >= bounding[0][0]) & (means[:, 0] <= bounding[0][3]) &
        (means[:, 1] >= bounding[0][1]) & (means[:, 1] <= bounding[0][4]) &
        (means[:, 2] >= bounding[0][2]) & (means[:, 2] <= bounding[0][5])
    )
    means = means[mask]
    shs = shs[mask]
    normals = normals[mask]
    poissonstart = time.time()
    Poisson_reconstruction(means,normals,shs)
    end = time.time()
    poissonend = time.time()
    poissondt = poissonend-poissonstart
    print(f"Poisson time: {end-start-tsdfdt}sec")
    #Den(means,shs)
    alpha_shape(means,shs)
    end = time.time()
    print(f"alpha time: {end-start-tsdfdt-poissondt}sec")
    """

    #"""
    start = time.time()
    means=model.gaussians["means"].cpu().numpy()
    scales=model.gaussians["scales"].cpu().numpy()
    rotations=model.gaussians["rotations"].cpu().numpy()
    opacities=model.gaussians["opacities"].cpu().numpy()
    shs = model.gaussians["shs"].detach().cpu().numpy()[:,0,:]
    means, opacities, scales, rotations, shs = rot_filter(means,opacities,scales, rotations, shs)
    us,vs,normals=quaternions_to_axes(rotations)
    #opacities = opa_norm(opacities)
    #means, opacities, scales, us, vs, normals, shs, gaussians_in_plane= gsfilter(means, opacities, scales, us, vs, normals, shs)
    means, opacities, scales, us, vs, normals, shs = gs_clean(means, opacities, scales, us, vs, normals, shs)
    for _ in range(1):
        means, opacities, scales, us, vs, normals, shs = gs_fill(means, opacities, scales, us, vs, normals, shs)
    #for _ in range(5):
        #means, opacities, scales, us, vs, normals, shs = gs_fill(means, opacities, scales, us, vs, normals, shs)

    name = 'fuse.ply'
    depth_trunc = args.depth_trunc
    voxel_size = args.voxel_size
    sdf_trunc = args.sdf_trunc

    #bounding=extract_bounding("data/geometry_gt/jinguilou_post/lidar.ply")
    bounding=extract_bounding_from_np(means=means)

    dx=(bounding[0][0]+bounding[0][3])/2-bounding[0][0]
    dy=(bounding[0][1]+bounding[0][4])/2-bounding[0][1]
    dz=(bounding[0][2]+bounding[0][5])/2-bounding[0][2]

    """
    bounding[0][0]+=0.85*dx
    bounding[0][3]-=0.6*dx
    bounding[0][1]+=0.85*dy
    bounding[0][4]-=0.6*dy
    bounding[0][2]+=0.85*dz
    bounding[0][5]-=0.6*dz"""

    """
    bounding[0][0]=0.186034-6 # truck
    bounding[0][3]=0.186034+6
    bounding[0][1]=0.092017-6
    bounding[0][4]=0.092017+6
    bounding[0][2]=0.098305-6
    bounding[0][5]=0.098305+6"""

    #"""
    bounding[0][0]=0.402157-15 # creepy
    bounding[0][3]=0.402157+15
    bounding[0][1]=1.066930-15
    bounding[0][4]=1.066930+15
    bounding[0][2]=-0.047880-15
    bounding[0][5]=-0.047880+15
    #"""

    """
    bounding[0][0]=2.445300-15 # museum
    bounding[0][3]=2.445300+15
    bounding[0][1]=-2.006680-15
    bounding[0][4]=-2.006680+15
    bounding[0][2]=4.817780-15
    bounding[0][5]=4.817780+15"""

    """
    bounding[0][0]=0.323853-15 # playroom
    bounding[0][3]=0.323853+15
    bounding[0][1]=0.106915-15
    bounding[0][4]=0.106915+15
    bounding[0][2]=0.420907-15
    bounding[0][5]=0.420907+15"""

    tsdf=TSDF_forGS.TSDF()
    tsdf.addGrids(bounding[0][0],bounding[0][1],bounding[0][2],bounding[0][3],bounding[0][4],bounding[0][5],voxel_size,sdf_trunc,depth_trunc)
    #for index in tqdm(range(cameras.__len__()),desc="TSDF Integrating"):
        #tsdf.TSDF_Integration(get_intrinsic(cameras[index]),get_extrinsic(cameras[index]),reds[index],greens[index],blues[index],depths[index][0],weights[index])
    
    for index in tqdm(range(len(means)),desc="Gaussian Integrating:"):
        tsdf.Gaussian_Integration(means[index].astype(np.float32),shs[index].astype(np.float32),normals[index].astype(np.float32),us[index].astype(np.float32),vs[index].astype(np.float32),scales[index].astype(np.float32),float(opacities[index]),1)
    points,triangles,colors=tsdf.extract_mesh()
    mesh=o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.vertex_colors=o3d.utility.Vector3dVector(np.clip(colors, 0, 1))
    mesh = filter_multiple(mesh, name)

    print("post-processing...")
    mesh_post = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
    o3d.io.write_triangle_mesh('fuse_post.ply', mesh_post)
    end = time.time()
    print(f"Ours time: {end-start}sec")
    #"""