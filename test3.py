import numpy as np
import utils


# naive implementation of the algorithm to create the data structure

# with dense voxels
voxels_s = [
    tuple([0, 0, 0]),
    tuple([0, 0, 1]),
]

voxels_d = utils.sparse_to_dense(voxels_s)


def ref_create_data_structure_from_dense(voxels):
    """

    Args:
        voxels: a 3D numpy array of shape (x, y, z) with boolean values
                True for occupied voxels and False for empty voxels.


    """
    ds = {}

    previous_plane = []
    for z in range(1, voxels.shape[2] - 1):
        this_plane = []
        for y in range(1, voxels.shape[1] - 1):
            for x in range(1, voxels.shape[0] - 1):
                # get the incident voxels
                index = np.array(x, y, z, dtype=np.int32)
                lll = tuple(index)
                llu = tuple(index + np.array([0, 0, 1], dtype=np.int32))
                lul = tuple(index + np.array([0, 1, 0], dtype=np.int32))
                luu = tuple(index + np.array([0, 1, 1], dtype=np.int32))
                ull = tuple(index + np.array([1, 0, 0], dtype=np.int32))
                ulu = tuple(index + np.array([1, 0, 1], dtype=np.int32))
                uul = tuple(index + np.array([1, 1, 0], dtype=np.int32))
                uuu = tuple(index + np.array([1, 1, 1], dtype=np.int32))

                # check if the incident voxels are all occupied or all empty
                # TODO do this in a more efficient way (with sum for example)
                if (
                    voxels[lll]
                    and voxels[llu]
                    and voxels[lul]
                    and voxels[luu]
                    and voxels[ull]
                    and voxels[ulu]
                    and voxels[uul]
                    and voxels[uuu]
                ) or (
                    not voxels[lll]
                    and not voxels[llu]
                    and not voxels[lul]
                    and not voxels[luu]
                    and not voxels[ull]
                    and not voxels[ulu]
                    and not voxels[uul]
                    and not voxels[uuu]
                ):
                    # ignore
                    continue
                # add the vertex to the data structure
                ds[index] = None
                this_plane.append(index)

            # we can now compute the neighbors of the previous z level
            for surface_vertex in previous_plane:
                # find a 6-connected neighbor of the surface vertex
                six_connected = None
                for dx in [-1, 1]:
                    n_index = tuple(x + dx, y, z)
                    if n_index in ds.keys():
                        six_connected = n_index
                if six_connected is None:
                    for dx in [-1, 1]:
                        n_index = tuple(x + dx, y, z)
                        if n_index in ds.keys():
                            six_connected = n_index
                if six_connected is None:
                    for dx in [-1, 1]:
                        n_index = tuple(x + dx, y, z)
                        if n_index in ds.keys():
                            six_connected = n_index
                if six_connected is None:
                    raise ValueError(
                        f"Could not find a 6-connected neighbor for the index {surface_vertex}"
                    )
                # from this neighbor, I need to cycle around the surface vertex to find the others
                # Ok I give up
            previous_plane = this_plane


def create_data_structure_from_dense(voxels):
    """
    My own reimplementation of the algorithm

    Args:
        voxels: a 3D numpy array of shape (x, y, z) with boolean values
                True for occupied voxels and False for empty voxels.

    Returns:
        ds: a dict which contain all surface vertexes as keys,
            and as values the set of faces associated to that corner
        faces: a set of faces, each face is a tuple of surface vertexes
    """

    FACES_LOOKUP = {
        "up": {
            "offset": np.array([0, 0, 1], dtype=np.int32),
            "corners": [
                np.array([0, 0, 1], dtype=np.int32),
                np.array([1, 0, 1], dtype=np.int32),
                np.array([1, 1, 1], dtype=np.int32),
                np.array([0, 1, 1], dtype=np.int32),
            ],
        },
        "down": {
            "offset": np.array([0, 0, -1], dtype=np.int32),
            "corners": [
                np.array([0, 0, 0], dtype=np.int32),
                np.array([1, 0, 0], dtype=np.int32),
                np.array([1, 1, 0], dtype=np.int32),
                np.array([0, 1, 0], dtype=np.int32),
            ],
        },
        "front": {
            "offset": np.array([0, -1, 0], dtype=np.int32),
            "corners": [
                np.array([0, 0, 0], dtype=np.int32),
                np.array([1, 0, 0], dtype=np.int32),
                np.array([1, 0, 1], dtype=np.int32),
                np.array([0, 0, 1], dtype=np.int32),
            ],
        },
        "back": {
            "offset": np.array([0, 1, 0], dtype=np.int32),
            "corners": [
                np.array([0, 1, 0], dtype=np.int32),
                np.array([1, 1, 0], dtype=np.int32),
                np.array([1, 1, 1], dtype=np.int32),
                np.array([0, 1, 1], dtype=np.int32),
            ],
        },
        "left": {
            "offset": np.array([-1, 0, 0], dtype=np.int32),
            "corners": [
                np.array([0, 0, 0], dtype=np.int32),
                np.array([0, 1, 0], dtype=np.int32),
                np.array([0, 1, 1], dtype=np.int32),
                np.array([0, 0, 1], dtype=np.int32),
            ],
        },
        "right": {
            "offset": np.array([1, 0, 0], dtype=np.int32),
            "corners": [
                np.array([1, 0, 0], dtype=np.int32),
                np.array([1, 1, 0], dtype=np.int32),
                np.array([1, 1, 1], dtype=np.int32),
                np.array([1, 0, 1], dtype=np.int32),
            ],
        },
    }

    ds = {}

    faces = set()
    for z in range(1, voxels.shape[2]):
        for y in range(1, voxels.shape[1]):
            for x in range(1, voxels.shape[0]):

                idx = np.array([x, y, z], dtype=np.int32)

                if not voxels[tuple(idx)]:
                    # discard empty voxels
                    continue

                for k, v in FACES_LOOKUP.items():

                    neighbor_idx = idx + v["offset"]
                    if voxels[tuple(neighbor_idx)]:
                        # face is not visible
                        continue

                    face = tuple([tuple(idx + corner) for corner in v["corners"]])
                    faces.add(face)

                    for corner in face:
                        ds[corner] = ds.get(corner, [])
                        ds[corner].append(face)

    return ds, faces


def initial_parametrization(corners, faces):

    def get_neighbors(corner, direct=True, sort=True):
        """
        Get the neighbors of a corner in the data structure.
        """

        neighbors = set()
        for face in corners[corner]:
            for c in face:
                neighbors.add(c)
        neighbors.discard(corner)
        if not direct:
            return neighbors

        # if direct, we only return the direct neighbors, that is the corners which have only one of the index different
        direct_neighbors = set()
        for n in neighbors:
            if sum([abs(n[i] - corner[i]) for i in range(3)]) == 1:
                direct_neighbors.add(n)

        neighbors = direct_neighbors

        if sort:
            sorted_neighbors = []
            # get the orientation of the faces
            neighbors = list(neighbors)
            normals = []
            angles = []
            for face in corners[corner]:
                normal = np.cross(
                    np.array(face[1]) - np.array(face[0]),
                    np.array(face[2]) - np.array(face[0]),
                )
                # get the normal that points outside the surface
                if tuple(corner + normal) in corners.keys():
                    normal = -normal
                normals.append(normal)
            # find the average normal
            average_normal = np.mean(normals, axis=0)
            average_normal /= np.linalg.norm(average_normal)
            # now corner and average_normal uniquely determine a plane

            # select a neighbor at random
            niter = iter(neighbors)
            first = next(niter)

            # project the neighbor onto the plane\
            d = np.array(first) - np.array(corner)
            projected_first = d - np.dot(d, average_normal) * average_normal
            angles.append(0.0)

            for c in niter:
                d = np.array(c) - np.array(corner)
                projected_c = d - np.dot(d, average_normal) * average_normal
                # compute the angle between projected_c and projected_first
                # TODO
        return neighbors

    # as of python 3.7, dicts are ordered by insertion order

    # find the two poles
    min_z = np.inf
    south = None
    max_z = -np.inf
    north = None
    for idx, corner in enumerate(corners.keys()):
        z = corner[2]
        if z < min_z:
            min_z = z
            south = idx
        elif z > max_z:
            max_z = z
            north = idx

    # build indexing and reverse indexing
    idx_corners = list(corners.keys())

    # re order the corners so that the north pole and the south pole are at the end
    # it makes it easier to build the linear system
    idx_corners[-2], idx_corners[north] = idx_corners[north], idx_corners[-2]
    idx_corners[-1], idx_corners[south] = idx_corners[south], idx_corners[-1]
    north = len(idx_corners) - 2
    south = len(idx_corners) - 1

    corner_idxs = {value: index for index, value in enumerate(idx_corners)}
    ##

    # Solve the initial latitude problem
    n_verts = len(idx_corners)
    A = np.zeros((n_verts - 2, n_verts - 2))
    b = np.zeros((n_verts - 2, 1))

    # I am currently taking into account all neighbors, not just the direct ones (like in the paper)

    for corner, idx in corner_idxs.items():
        if idx in [north, south]:
            continue
        neighbors = set([corner_idxs[c] for c in get_neighbors(corner)])

        if south in neighbors:
            b[idx] = np.pi

        neighbors.discard(idx)

        A[idx, idx] = len(neighbors)

        neighbors.discard(south)
        neighbors.discard(north)

        A[idx, tuple(neighbors)] = -1

    # A is symmetric and sparse, but I do not care for now
    lats = np.linalg.solve(A, b).flatten()

    # add the boundary conditions for the north and south poles
    lats = np.append(lats, [0.0, np.pi])

    latitudes = {idx_corners[i]: lats[i] for i in range(len(idx_corners))}

    # Solve the initial longitude problem

    # find the line
    north_corner = idx_corners[north]
    south_corner = idx_corners[south]

    # now modify matrix A and vector b
    north_neighbors_idx = [corner_idxs[n] for n in get_neighbors(north_corner)]
    south_neighbors_idx = [corner_idxs[s] for s in get_neighbors(south_corner)]
    A[north_neighbors_idx, north_neighbors_idx] -= 1.0
    A[south_neighbors_idx, south_neighbors_idx] -= 1.0
    # add the boundary conditions to an arbitrary point
    A[0, 0] += 2.0

    b[:] = 0.0

    previous = north
    here = north_neighbors_idx[0]
    maximum = 0.0

    while here != south:
        neighbors_idx = [corner_idxs[n] for n in get_neighbors(idx_corners[here])]
        for n_idx in neighbors_idx:
            lat = lats[n_idx]
            if lat > maximum:
                maximum = lat
                next_here = n_idx
            if n_idx == previous:
                prev_pos = n_idx
        # I have now found a line. Points to the left get -2pi, points to the right get +2pi
        # Now I have to be careful and use actual direct neighbors
        for n_idx in neighbors_idx:
            if n_idx == next_here:
                continue
            if n_idx == prev_pos:
                continue
            n_coords = idx_corners[n_idx]
        break

    return latitudes


corners, visible_faces = create_data_structure_from_dense(voxels_d)

lats = initial_parametrization(corners, visible_faces)

import pyvista as pv
import numpy as np

# Initialize point list and face list
points = []
faces = []
scalars = []


# Helper to add a square given its origin and size
def add_square(corners, scalar_values):
    """
    Add a square to the global mesh, given its 4 corners and scalar values.

    Parameters:
    - corners: list or array of shape (4, 3), each row is [x, y, z]
    - scalar_values: list of 4 scalar values corresponding to each corner
    """
    assert len(corners) == 4, "Exactly 4 corner points are required."
    assert len(scalar_values) == 4, "Exactly 4 scalar values are required."

    idx = len(points)  # base index for this square

    # Append points and scalars
    points.extend(corners)
    scalars.extend(scalar_values)

    # Define the face using point indices
    faces.extend([4, idx, idx + 1, idx + 2, idx + 3])


for face in visible_faces:
    lats_face = [lats[corner] for corner in face]

    add_square(
        corners=[list(c) for c in face],
        scalar_values=lats_face,
    )
# Convert to numpy arrays
points_np = np.array(points)
faces_np = np.array(faces)
scalars_np = np.array(scalars)

# Create the mesh
mesh = pv.PolyData(points_np, faces_np)
mesh.point_data["scalars"] = scalars_np

# Plot the mesh with interpolated scalar coloring
plotter = pv.Plotter()
plotter.add_mesh(
    mesh,
    scalars="scalars",
    cmap="viridis",
    show_edges=True,
    interpolate_before_map=True,
)
plotter.show()
