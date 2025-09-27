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
from internal.geometry.gsfilter import gsfilter, gs_fill, gs_plane, gs_clean
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
    loadable_file = "/media/allen/新加卷/CityGaussian/outputs/jinguilou_post/checkpoints/epoch=811-step=30000.ckpt"
    #loadable_file = "/media/allen/新加卷/CityGaussian/outputs/jinguilou_post/checkpoints/epoch=28-step=1000.ckpt"

    print(loadable_file)
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

    # load dataset
    """
    dataparser_outputs = dataparser_config.instantiate(#用数据路径等参数初始化
        path=dataset_path,
        output_path=os.getcwd(),
        global_rank=0,
    ).get_outputs()#处理好相片、深度对齐的数据
    cameras = [i.to_device(device) for i in dataparser_outputs.train_set.cameras]"""

    """
    cameras=add_cameras("/media/allen/新加卷/CityGaussian/data/geometry_gt/jinguilou_post/lidar.ply",cameras,device)
    for camera in cameras:
        camera.idx.to(device)
        camera.to_device(device)"""
    #visualising_cameras(cameras,"/media/allen/新加卷/CityGaussian/data/geometry_gt/jinguilou_post/lidar.ply")

    # set the active_sh to 0 to export only diffuse texture
    #model.active_sh_degree = 0#只保留基础球谐函数
    #bg_color = torch.zeros((3,), dtype=torch.float, device=device)#背景颜色
    #maps = GS2DMeshingUtils.render_views(model, renderer, cameras, bg_color)#再一次渲染得到的深度图、rgb图
    #colors,depths,weights=maps
    #reds=[color[0,:,:] for color in colors]
    #greens=[color[1,:,:] for color in colors]
    #blues=[color[2,:,:] for color in colors]

    means=model.gaussians["means"].cpu().numpy()
    scales=model.gaussians["scales"].cpu().numpy()
    rotations=model.gaussians["rotations"].cpu().numpy()
    us,vs,normals=quaternions_to_axes(rotations)
    opacities=model.gaussians["opacities"].cpu().numpy()
    shs = model.gaussians["shs"].detach().cpu().numpy()[:,0,:]
    #means, opacities, scales, us, vs, normals, shs, gaussians_in_plane= gsfilter(means, opacities, scales, us, vs, normals, shs)
    means, opacities, scales, us, vs, normals, shs = gs_clean(means, opacities, scales, us, vs, normals, shs)
    for _ in range(2):
        means, opacities, scales, us, vs, normals, shs = gs_fill(means, opacities, scales, us, vs, normals, shs)
    gaussians_in_plane = gs_plane(means, opacities, scales, us, vs, normals, shs)
    #for _ in range(5):
        #means, opacities, scales, us, vs, normals, shs = gs_fill(means, opacities, scales, us, vs, normals, shs)

    name = 'fuse.ply'
    depth_trunc = args.depth_trunc
    voxel_size = args.voxel_size
    sdf_trunc = args.sdf_trunc
    bounding=extract_bounding("data/geometry_gt/jinguilou_post/lidar.ply")
    #bounding=extract_bounding_from_np(means=means)

    tsdf=TSDF_forGS.TSDF()
    tsdf.addGrids(bounding[0][0],bounding[0][1],bounding[0][2],bounding[0][3],bounding[0][4],bounding[0][5],voxel_size,sdf_trunc,depth_trunc)
    #for index in tqdm(range(cameras.__len__()),desc="TSDF Integrating"):
        #tsdf.TSDF_Integration(get_intrinsic(cameras[index]),get_extrinsic(cameras[index]),reds[index],greens[index],blues[index],depths[index][0],weights[index])
    
    for index in tqdm(range(len(means)),desc="Gaussian Integrating:"):
        if gaussians_in_plane[index]:
            tsdf.Gaussian_Integration(means[index].astype(np.float32),shs[index].astype(np.float32),normals[index].astype(np.float32),us[index].astype(np.float32),vs[index].astype(np.float32),scales[index].astype(np.float32),float(opacities[index]),1.5)
        else:
            tsdf.Gaussian_Integration(means[index].astype(np.float32),shs[index].astype(np.float32),normals[index].astype(np.float32),us[index].astype(np.float32),vs[index].astype(np.float32),scales[index].astype(np.float32),float(opacities[index]),1)
    points,triangles,colors=tsdf.extract_mesh()
    mesh=o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.vertex_colors=o3d.utility.Vector3dVector(colors)
    filter_multiple(mesh, name)