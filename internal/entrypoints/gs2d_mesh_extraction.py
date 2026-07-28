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
from internal.geometry.gsfilter import gsfilter, gs_fill, gs_plane, gs_clean, opa_norm, rot_filter, gs_fill_pca, gs_fill_ransac
from internal.geometry.filter import filter_multiple
from internal.utils.sh_utils import SH2RGB
from scipy.spatial import Delaunay
import time
from scipy.spatial import KDTree
import psutil
import os
import time
import threading
process = psutil.Process(os.getpid())
peak_mem = 0
stop_monitor = False
def monitor():
    global peak_mem
    while not stop_monitor:
        mem = process.memory_info().rss / 1024**3
        peak_mem = max(peak_mem, mem)
        time.sleep(0.1)  #  0.1s 

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

    print("Poission")
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=7)

    #  KDTree 
    if colors is not None:
        print(" KDTree ...")
        #  KDTree
        pcd_tree = o3d.geometry.KDTreeFlann(point_cloud)
        kdtree = KDTree(means)
        mesh_colors = []
        mesh_vertices = np.asarray(mesh.vertices)

        _, idx = kdtree.query(mesh_vertices, k=1)              # 
        mesh_colors = colors[idx]

        mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(mesh_colors))

    o3d.io.write_triangle_mesh("poisson_reconstructed.ply", mesh)

def Den(means: np.ndarray, shs: np.ndarray = None):
    colors = SH2RGB(shs)
    # 3D Delaunay 
    print("Denaulay")
    tri = Delaunay(means)

    faces = []
    for tet in tri.simplices:
        #  4 
        for i in range(4):
            face = np.delete(tet, i)
            faces.append(face)

    faces = np.array(faces)

    # === ===
    faces_sorted = np.sort(faces, axis=1)
    faces_unique = np.unique(faces_sorted, axis=0)

    #  Open3D 
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(means)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    if colors is not None:
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_triangle_mesh("delaunay_surface_mesh.ply", mesh)

def alpha_shape(means: np.ndarray, shs: np.ndarray = None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(means)
    colors = SH2RGB(shs)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    #pcd_dense = densify_point_cloud(pcd, radius=1, n_samples=5)
    
    # === Alpha Shape  ===
    print(f" Alpha Shape  (alpha={5}) ...")
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, 0.03)
    mesh.compute_vertex_normals()

    if colors is not None:
        print(" ...")
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        mesh_colors = []
        for v in np.asarray(mesh.vertices):
            [_, idx, _] = pcd_tree.search_knn_vector_3d(v, 1)  # 1
            mesh_colors.append(colors[idx[0]])
        mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)
    
    # ===  ===
    o3d.io.write_triangle_mesh("alpha_surface_mesh.ply", mesh)


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
    t = threading.Thread(target=monitor, daemon=True)
    t.start()
    args = CLI(CLIArgs)

    device = torch.device("cuda")

    # load ckpt
    loadable_file = GaussianModelLoader.search_load_file(args.model_path)

    print(loadable_file)
    start = time.time()
    dataparser_config = None
    if loadable_file.endswith(".ckpt"):#ckpt
        ckpt = torch.load(loadable_file, map_location="cpu")#GPU
        # initialize model
        model = GaussianModelLoader.initialize_model_from_checkpoint(
            ckpt,
            device=device,
        )
        #ckptcpu

        model.freeze()
        model.pre_activate_all_properties()#

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
    
    dataparser_outputs = dataparser_config.instantiate(#
        path=dataset_path,
        output_path=os.getcwd(),
        global_rank=0,
    ).get_outputs()#
    cameras = [i.to_device(device) for i in dataparser_outputs.train_set.cameras]

    # set the active_sh to 0 to export only diffuse texture
    model.active_sh_degree = 0#
    bg_color = torch.zeros((3,), dtype=torch.float, device=device)#
    maps = GS2DMeshingUtils.render_views(model, renderer, cameras, bg_color)#rgb
    bound = GS2DMeshingUtils.estimate_bounding_sphere(cameras)

    
    _, radius = bound
    depth_trunc = (radius * 2.0) if args.depth_trunc < 0 else args.depth_trunc
    voxel_size = (depth_trunc / args.mesh_res) if args.voxel_size < 0 else args.voxel_size
    sdf_trunc = 5.0 * voxel_size if args.sdf_trunc < 0 else args.sdf_trunc
    mesh = GS2DMeshingUtils.extract_mesh_bounded(maps=maps, cameras=cameras, voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)
    output_dir = args.model_path
    name = f'fuse_{voxel_size}.ply'
    if os.path.isfile(output_dir):
        output_dir = os.path.dirname(output_dir)
    o3d.io.write_triangle_mesh(os.path.join(output_dir, name), mesh)
    end = time.time()
    tsdfend = time.time()
    tsdfdt = tsdfend-tsdfstart
    print(f"TSDF time: {end-start}sec")
    print("mesh saved at {}".format(os.path.join(output_dir, name)))
    """
    """
    means=model.gaussians["means"].cpu().numpy()
    scales=model.gaussians["scales"].cpu().numpy()
    rotations=model.gaussians["rotations"].cpu().numpy()
    opacities=model.gaussians["opacities"].cpu().numpy()
    shs = model.gaussians["shs"].detach().cpu().numpy()[:,0,:]
    means, opacities, scales, rotations, shs = rot_filter(means,opacities,scales, rotations, shs)
    us,vs,normals=quaternions_to_axes(rotations)
    poissonstart = time.time()
    Poisson_reconstruction(means,normals,shs)
    end = time.time()
    poissonend = time.time()
    poissondt = poissonend-poissonstart
    #print(f"Poisson time: {end-start-tsdfdt}sec")
    #Den(means,shs)
    alpha_shape(means,shs)
    end = time.time()
    #print(f"alpha time: {end-start-tsdfdt-poissondt}sec")
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
    means, opacities, scales, us, vs, normals, shs = gs_clean(means, opacities, scales, us, vs, normals, shs)
    for _ in range(1):
        #pass
        means, opacities, scales, us, vs, normals, shs = gs_fill(means, opacities, scales, us, vs, normals, shs)
        #means, opacities, scales, us, vs, normals, shs = gs_fill_ransac(means, opacities, scales, us, vs, normals, shs)
        #means, opacities, scales, us, vs, normals, shs = gs_fill_pca(means, opacities, scales, us, vs, normals, shs)

    name = 'fuse'
    depth_trunc = args.depth_trunc
    voxel_size = args.voxel_size
    sdf_trunc = args.sdf_trunc
    name += f"{voxel_size:.2f}.ply"

    bounding=extract_bounding_from_np(means=means)

    tsdf=TSDF_forGS.TSDF()
    tsdf.addGrids(bounding[0][0],bounding[0][1],bounding[0][2],bounding[0][3],bounding[0][4],bounding[0][5],voxel_size,sdf_trunc,depth_trunc)
    
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
    print(f"Extraction time: {end-start}sec")
    #"""
    
    """
    log_file = "timing.txt"
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            lines = f.readlines()
    else:
        lines = []
    with open(log_file, 'a') as f:
        f.write(f"{voxel_size}\t{end - start:.2f}\n")
    print(f" {log_file}: voxelsize={voxel_size}, time={end - start:.2f}s")
    stop_monitor = True
    t.join(timeout=1)
    print(f": {peak_mem:.2f} GB")
    """