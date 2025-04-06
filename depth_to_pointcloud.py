import numpy as np
import cv2
import open3d as o3d

def depth_to_point_cloud(image_path, depth_path):
    img = cv2.imread(image_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)

    h, w = depth.shape
    fx = fy = 1.0
    cx, cy = w / 2, h / 2

    points = []
    colors = []

    for y in range(h):
        for x in range(w):
            z = depth[y, x] / 255.0
            X = (x - cx) * z / fx
            Y = (y - cy) * z / fy
            points.append([X, -Y, -z])  # Flip Y and Z for visualization
            colors.append(img[y, x] / 255.0)

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(np.array(points))
    pc.colors = o3d.utility.Vector3dVector(np.array(colors))

    o3d.io.write_point_cloud("output.ply", pc)
    o3d.visualization.draw_geometries([pc])

    print("Point cloud saved as output.ply")

# Run it
if __name__ == "__main__":
    depth_to_point_cloud("input.jpg", "depth_map.png")
