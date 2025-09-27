import open3d as o3d

def filter_multiple(mesh,meshfile):
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    o3d.io.write_triangle_mesh(meshfile,mesh)

if __name__=="__main__":
    filter_multiple("/media/allen/新加卷/CityGaussian/fuse.ply")