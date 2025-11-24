import numpy as np
import open3d as o3d


def superquadric_mesh(a=1, b=1, c=1, epsilon1=1, epsilon2=1, resolution=100):
    el = np.linspace(-np.pi / 2, np.pi / 2, resolution)
    az = np.linspace(-np.pi, np.pi, resolution)
    el, az = np.meshgrid(el, az)

    def sgn_pow(x, e):
        return np.sign(x) * (np.abs(x) ** e)

    # Parametric equations
    x = a * sgn_pow(np.cos(el), epsilon1) * sgn_pow(np.cos(az), epsilon2)
    y = b * sgn_pow(np.cos(el), epsilon1) * sgn_pow(np.sin(az), epsilon2)
    z = c * sgn_pow(np.sin(el), epsilon1)

    # Flatten and stack vertices
    vertices = np.c_[x.flatten(), y.flatten(), z.flatten()]

    # Construct faces (triangles) from grid indices
    faces = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            idx = lambda u, v: u * resolution + v
            # each face is made up by two triangles
            faces.append([idx(i, j), idx(i + 1, j), idx(i, j + 1)])
            faces.append([idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    return mesh
