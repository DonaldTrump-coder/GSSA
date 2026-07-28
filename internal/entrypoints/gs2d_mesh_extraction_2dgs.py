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

def extract_bounding(ply_path:str):
    ply=o3d.io.read_point_cloud(ply_path)
    xmin,ymin,zmin=ply.get_min_bound()
    xmax,ymax,zmax=ply.get_max_bound()
    return np.array([[xmin,ymin,zmin,xmax,ymax,zmax]],dtype=np.float64)

def extract_bounding_from_np(means):
    xmin,ymin,zmin=means.min(axis=0)
    xmax,ymax,zmax=means.max(axis=0)
    return np.array([[xmin,ymin,zmin,xmax,ymax,zmax]],dtype=np.float64)

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
    loadable_file = os.path.join(args.model_path,"checkpoints","chkpnt30000.pth")

    print(loadable_file)
    state_dict = torch.load(loadable_file, map_location='cpu', weights_only=False)
    
    tensors = [t for t in state_dict[0] if torch.is_tensor(t)]

    #"""
    means=tensors[0].detach().cpu().numpy()
    shs = tensors[1].detach().cpu().numpy()[:,0,:]
    scales=torch.exp(tensors[3].detach()).cpu().numpy()
    rotations=torch.nn.functional.normalize(tensors[4].detach()).cpu().numpy()
    opacities=torch.sigmoid(tensors[5].detach()).cpu().numpy()
    means, opacities, scales, rotations, shs = rot_filter(means,opacities,scales, rotations, shs)
    us,vs,normals=quaternions_to_axes(rotations)
    opacities = opa_norm(opacities)
    #means, opacities, scales, us, vs, normals, shs, gaussians_in_plane= gsfilter(means, opacities, scales, us, vs, normals, shs)
    means, opacities, scales, us, vs, normals, shs = gs_clean(means, opacities, scales, us, vs, normals, shs)
    for _ in range(1):
        means, opacities, scales, us, vs, normals, shs = gs_fill(means, opacities, scales, us, vs, normals, shs)

    name = 'fuse.ply'
    depth_trunc = args.depth_trunc
    voxel_size = args.voxel_size
    sdf_trunc = args.sdf_trunc

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
    #"""