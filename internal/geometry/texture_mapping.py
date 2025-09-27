import open3d as o3d
from internal.cameras import Camera
import numpy as np
from internal.geometry.filter import filter_multiple

def mapping(meshfile,cameras:list[Camera],colors):
    filter_multiple(meshfile=meshfile)
    mesh=o3d.io.read_triangle_mesh(meshfile)
    tmesh=o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    tmesh=tmesh.remove_non_manifold_edges()
    tmesh=tmesh.remove_unreferenced_vertices()
    tmesh.compute_uvatlas()
    intrinsics=[o3d.core.Tensor(np.array([
                [camera.fx.cpu(), 0, camera.cx.cpu()],
                [0, camera.fy.cpu(), camera.cy.cpu()],
                [0, 0, 1]
                ], dtype=np.float64)) for camera in cameras]
    extrinsics=[o3d.core.Tensor(np.vstack((np.hstack((np.asarray(camera.R.cpu(), dtype=np.float64).reshape(3, 3),
                                    np.asarray(camera.T.cpu(), dtype=np.float64).reshape(3, 1))),
                          np.array([0,0,0,1], dtype=np.float64)))) for camera in cameras]
    images=[o3d.t.geometry.Image(image.numpy()) for image in colors]
    albedo=tmesh.project_images_to_albedo(images,intrinsics,extrinsics,tex_size=1024)
    tmesh.material.texture_maps["albedo"]=albedo
    mesh=tmesh.to_legacy()
    o3d.io.write_triangle_mesh("data/fuse_with_color.ply",mesh)