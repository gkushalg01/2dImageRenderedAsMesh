import open3d as o3d
import numpy as np

def mesh_from_point_cloud(ply_file: str, output_file: str = "output_mesh.ply"):
    # Load point cloud
    pcd = o3d.io.read_point_cloud(ply_file)

    # Estimate and orient normals
    print("[INFO] Estimating normals...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(30)

    # Poisson reconstruction
    print("[INFO] Running Poisson surface reconstruction...")
    try:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
    except Exception as e:
        print(f"[ERROR] Poisson reconstruction failed: {e}")
        return

    # Remove low-density vertices
    print("[INFO] Removing low-density artifacts...")
    densities = np.asarray(densities)
    threshold = np.quantile(densities, 0.05)
    vertices_to_keep = densities > threshold
    mesh.remove_vertices_by_mask(~vertices_to_keep)

    # Save and show mesh
    o3d.io.write_triangle_mesh(output_file, mesh)
    print(f"[INFO] Saved mesh to {output_file}")
    o3d.visualization.draw_geometries([mesh])

if __name__ == "__main__":
    mesh_from_point_cloud("output.ply")

    #  current implementation is not enough.
