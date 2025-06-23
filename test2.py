import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# # 1. Load ShapeNetCore mesh
# dataset = ShapeNetCore(
#     "/home/fabioscap/Desktop/sq/data", synsets=["airplane"], version=2
# )
# mesh_data = dataset[1]
# verts = mesh_data["verts"]
# faces = mesh_data["faces"]

# verts_np = verts.cpu().numpy()
# faces_np = faces.cpu().numpy()

# mesh = o3d.geometry.TriangleMesh()
# mesh.vertices = o3d.utility.Vector3dVector(verts_np)
# mesh.triangles = o3d.utility.Vector3iVector(faces_np)
# mesh.compute_vertex_normals()

# # 2. Create voxel grid
# voxel_size = 0.05
# voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(
#     mesh, voxel_size=voxel_size
# )


# # 3. Construct wireframe cubes for each voxel
# def create_voxel_wireframe(center, size):
#     d = size / 2.0
#     offsets = np.array(
#         [
#             [-d, -d, -d],
#             [d, -d, -d],
#             [d, d, -d],
#             [-d, d, -d],
#             [-d, -d, d],
#             [d, -d, d],
#             [d, d, d],
#             [-d, d, d],
#         ]
#     )
#     corners = offsets + center

#     lines = [
#         [0, 1],
#         [1, 2],
#         [2, 3],
#         [3, 0],  # bottom square
#         [4, 5],
#         [5, 6],
#         [6, 7],
#         [7, 4],  # top square
#         [0, 4],
#         [1, 5],
#         [2, 6],
#         [3, 7],  # vertical edges
#     ]
#     return corners, lines


# def plot_voxel_grid(voxel_grid):
#     all_points = []
#     all_lines = []
#     colors = []
#     line_index = 0

#     for voxel in voxel_grid.get_voxels():
#         center = (
#             voxel.grid_index * voxel_grid.voxel_size
#             + voxel_grid.origin
#             + voxel_grid.voxel_size / 2
#         )
#         corners, lines = create_voxel_wireframe(center, voxel_grid.voxel_size)
#         all_points.extend(corners)
#         all_lines.extend([[i + line_index, j + line_index] for i, j in lines])
#         line_index += 8
#         colors.extend([[1, 0, 0]] * len(lines))  # black lines

#     # Create LineSet for wireframe
#     wireframe = o3d.geometry.LineSet()
#     wireframe.points = o3d.utility.Vector3dVector(np.array(all_points))
#     wireframe.lines = o3d.utility.Vector2iVector(np.array(all_lines))
#     wireframe.colors = o3d.utility.Vector3dVector(np.array(colors))

#     o3d.visualization.draw_geometries([voxel_grid, wireframe])


def get_neighbors(voxel_grid, voxel_index):
    # TODO cache this
    indexes = set(tuple(voxel.grid_index) for voxel in voxel_grid.get_voxels())

    voxel_index = np.array(voxel_index)
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                neighbor_index = voxel_index + np.array([dx, dy, dz])
                if tuple(neighbor_index) in indexes:
                    neighbors.append(neighbor_index)
    return neighbors


def set_axes_equal(ax):
    """Set 3D plot axes to have equal scale."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


# could save a lot of time merging these two functions
def find_vertexes(voxels):

    vertexes = set()
    for voxel in voxels:
        lll = tuple(voxel)
        llu = tuple(voxel + np.array([0, 0, 1], dtype=np.int32))
        lul = tuple(voxel + np.array([0, 1, 0], dtype=np.int32))
        luu = tuple(voxel + np.array([0, 1, 1], dtype=np.int32))
        ull = tuple(voxel + np.array([1, 0, 0], dtype=np.int32))
        ulu = tuple(voxel + np.array([1, 0, 1], dtype=np.int32))
        uul = tuple(voxel + np.array([1, 1, 0], dtype=np.int32))
        uuu = tuple(voxel + np.array([1, 1, 1], dtype=np.int32))
        vertexes = vertexes.union([lll, llu, lul, luu, ull, ulu, uul, uuu])

    return vertexes


def find_neighbors(vertexes):
    # each vertex has a maximum of 26 neighbors
    neighbors = []
    for vertex in vertexes:
        neighbors.append(set())
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    neighbor_index = tuple(vertex + np.array([dx, dy, dz]))
                    if neighbor_index in vertexes:
                        neighbors[-1].add(neighbor_index)
    return neighbors


def find_inside_mask(neighbors):
    return [len(n) == 26 for n in neighbors]


def plot(voxels, vertexes, ax=None):
    # Draw each vertex in vertex_coordinates in 3D using matplotlib
    neighbors = find_neighbors(vertexes)
    inside_mask = find_inside_mask(neighbors)
    vertexes = np.array(list(vertexes))
    voxels = np.array(list(voxels))

    outside_mask = [not i for i in inside_mask]
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    # Plot each vertex as a green point
    ax.scatter(
        vertexes[outside_mask, 0],
        vertexes[outside_mask, 1],
        vertexes[outside_mask, 2],
        c="g",
        marker="o",
    )
    ax.scatter(
        vertexes[inside_mask, 0],
        vertexes[inside_mask, 1],
        vertexes[inside_mask, 2],
        c="orange",
        marker="o",
    )

    ax.scatter(
        voxels[:, 0] + 0.5,
        voxels[:, 1] + 0.5,
        voxels[:, 2] + 0.5,
        c="k",
        marker="x",
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    set_axes_equal(ax)

    return ax


# voxels = set(tuple(voxel.grid_index) for voxel in voxel_grid.get_voxels())
# Sparse representation vs Dense representation

voxels = set(
    [
        tuple([0, 0, 0]),
        tuple([0, 0, 1]),
        tuple([0, 1, 0]),
        tuple([0, 1, 1]),
        tuple([1, 0, 0]),
        tuple([1, 0, 1]),
        tuple([1, 1, 0]),
        tuple([1, 1, 1]),
    ]
)

vertexes = find_vertexes(voxels)
neighbors = find_neighbors(vertexes)
inside_mask = find_inside_mask(neighbors)


plot(voxels, vertexes)

plt.show()
