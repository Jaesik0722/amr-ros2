"""Lidar map -> collision / lidar mesh for the Isaac Sim twin (USDA, z up, metres).

    ros2 run amr_capture pcd_to_mesh MAP.pcd OUT.usda [--alignment DATASET/alignment.txt]
        [--voxel 0.03] [--depth 10] [--density-quantile 0.05]

Poisson surface reconstruction (Open3D, `pip install open3d`) of the FAST-LIO map, in the
same frame as the Gaussian Splatting dataset when --alignment is given. Load it with
run_sim.sh --env-usd <splats>.usdz --env-collision OUT.usda.
"""

import argparse

import numpy as np

from .export_dataset import read_pcd, transform_points


def write_usda(path, vertices, triangles):
    pts = ','.join(f'({x:.4f},{y:.4f},{z:.4f})' for x, y, z in vertices)
    idx = ','.join(str(int(i)) for i in np.asarray(triangles).ravel())
    with open(path, 'w') as f:
        f.write('#usda 1.0\n(\n    defaultPrim = "site"\n    upAxis = "Z"\n'
                '    metersPerUnit = 1\n)\n\ndef Xform "site"\n{\n'
                '    def Mesh "collision"\n    {\n'
                '        uniform token subdivisionScheme = "none"\n'
                f'        int[] faceVertexCounts = [{",".join(["3"] * len(triangles))}]\n'
                f'        int[] faceVertexIndices = [{idx}]\n'
                f'        point3f[] points = [{pts}]\n'
                '        color3f[] primvars:displayColor = [(0.6, 0.6, 0.6)]\n'
                '    }\n}\n')


def reconstruct(points, voxel=0.03, depth=10, density_quantile=0.05):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd = pcd.voxel_down_sample(voxel)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 4, max_nn=30))
    # Normals point into the scanned (indoor) space: towards a viewpoint at the middle of
    # the map, 1 m above the floor. Much faster than a consistent tangent-plane orientation.
    centre = pcd.get_axis_aligned_bounding_box().get_center()
    pcd.orient_normals_towards_camera_location(np.array([centre[0], centre[1], 1.0]))
    mesh, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    density = np.asarray(density)
    mesh.remove_vertices_by_mask(density < np.quantile(density, density_quantile))
    # Poisson closes the volume; trim it back to the scanned extent.
    mesh = mesh.crop(pcd.get_axis_aligned_bounding_box())
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    return np.asarray(mesh.vertices), np.asarray(mesh.triangles)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('pcd')
    ap.add_argument('out')
    ap.add_argument('--alignment', help='alignment.txt from export_dataset')
    ap.add_argument('--voxel', type=float, default=0.03)
    ap.add_argument('--depth', type=int, default=10)
    ap.add_argument('--density-quantile', type=float, default=0.05)
    a = ap.parse_args(argv)
    points = read_pcd(a.pcd)
    if a.alignment:
        points = transform_points(np.loadtxt(a.alignment), points)
    vertices, triangles = reconstruct(points, a.voxel, a.depth, a.density_quantile)
    write_usda(a.out, vertices, triangles)
    print(f'{len(triangles)} triangles -> {a.out}')


if __name__ == '__main__':
    main()
