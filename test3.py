import numpy as np
import utils


# naive implementation of the algorithm to create the data structure

# with dense voxels
voxels_s = [
    tuple([0, 0, 0]),
    tuple([1, 0, 0]),
    tuple([2, 0, 0]),
    tuple([0, 1, 0]),
    tuple([1, 1, 0]),
    tuple([2, 1, 0]),
    tuple([1, 1, 1]),
    tuple([1, 1, 2]),
    tuple([1, 2, 2]),
]

# voxels_s = [
#     tuple([0, 0, 0]),
#     tuple([1, 0, 0]),
# ]

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

    def get_neighbors(corner: np.ndarray, direct=True, sort=True):
        """
        Get the neighbors of a corner in the data structure.

        Args:
            corner: a numpy array of shape (3,) representing the corner
            direct: if True, only return the direct neighbors (they share an edge)
            sort: if True, sort the neighbors clockwise around the corner
        Returns:
            neighbors: a container of corners that are neighbors of the given corner.
                       If sorted is False, it is a set, otherwise it is a list.

        """

        # print(f"Corner: {corner}")

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

        # print(f"Neighbors before sorting: {neighbors}")

        if sort:
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

            # print(f"Average normal: {average_normal}")

            # select a neighbor at random
            niter = iter(neighbors)
            first = next(niter)

            # project the neighbor onto the plane\
            d = np.array(first) - np.array(corner)
            # print(f" d: {d}")
            projected_first = d - np.dot(d, average_normal) * average_normal
            # print(f"Projected first: {projected_first}")
            projected_first = projected_first / np.linalg.norm(projected_first)
            # complete the plane basis
            orth = np.cross(average_normal, projected_first)
            # print(f"Basis vectors: {projected_first}, {orth}")
            angles.append(0.0)

            for c in niter:
                # print(f"Processing neighbor: {c}")
                d = np.array(c) - np.array(corner)
                projected_c = d - np.dot(d, average_normal) * average_normal

                x = np.dot(projected_c, projected_first)
                y = np.dot(projected_c, orth)
                angle = np.arctan2(y, x)
                # wrap angle to [0, 2*pi]
                if angle < 0:
                    angle += 2 * np.pi
                # print(f"Angle for neighbor {c}: {angle}")
                angles.append(angle)

            # sort the neighbors by angle
            neighbors = [x for _, x in sorted(zip(angles, neighbors), reverse=True)]

        # print(f"Neighbors after sorting: {neighbors}")

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
    # print("BRUTE FORCING THE POLES, PLEASE REMOVE THIS")
    # for idx, corner in enumerate(corners.keys()):
    #     if corner == (1, 1, 1):
    #         north = idx
    #     elif corner == (3, 2, 2):
    #         south = idx

    # build indexing and reverse indexing
    idx_corners = list(corners.keys())
    # re order the corners so that the north pole and the south pole are at the end
    # it makes it easier to build the linear system
    # idx_corners[-2], idx_corners[north] = idx_corners[north], idx_corners[-2]
    # idx_corners[-1], idx_corners[south] = idx_corners[south], idx_corners[-1]
    # north = len(idx_corners) - 2
    # south = len(idx_corners) - 1
    # Ensure you handle ordering: north comes before south in the final list
    north_val = idx_corners[north]
    south_val = idx_corners[south]

    # Remove both first (remove the higher index first to avoid reindexing issues)
    for i in sorted([north, south], reverse=True):
        idx_corners.pop(i)

    # Append them to the end
    idx_corners.append(north_val)
    idx_corners.append(south_val)

    # Update indices
    north = len(idx_corners) - 2
    south = len(idx_corners) - 1

    corner_idxs = {value: index for index, value in enumerate(idx_corners)}
    ##

    # Solve the initial latitude problem
    n_verts = len(idx_corners)
    A = np.zeros((n_verts - 2, n_verts - 2))
    b = np.zeros((n_verts - 2, 1))

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

    # print(A)
    # print(b)

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

    prev_pos = north
    here = north_neighbors_idx[0]

    # print("BRUTE FORCING here, PLEASE REMOVE")
    # here = 0

    maximum = -1
    # print("North and south")
    # print(north)
    # print(south)
    while here != south:

        # print(f"Current point: {here}, previous: {prev_pos}")
        neighbors_idx = [
            corner_idxs[n]
            for n in get_neighbors(idx_corners[here], sort=True, direct=True)
        ]
        # print(neighbors_idx)
        for n_idx in neighbors_idx:
            lat = lats[n_idx]
            # print(f"latitude of neighbor {n_idx}: {lat}")
            if lat > maximum:
                maximum = lat
                next_here = n_idx

        # if here == 1:
        #     print("BRUTEFORCING next for node 1")
        #     next_here = 4

        # print(f"prev_pos: {prev_pos}, next_here: {next_here}")
        # print(f"neighbors: {neighbors_idx}")

        # iterate from prev_pos to next_here
        i = (neighbors_idx.index(prev_pos) - 1) % len(neighbors_idx)

        stop = neighbors_idx.index(next_here)
        # print(neighbors_idx)
        # print(f"stop {stop}")
        # loop from i+1 till stop handling eventual wrap around
        while i != stop:
            # print(i)
            n_idx = neighbors_idx[i]
            # print(n_idx)
            # print(f"decrementing {here}, incrementing {n_idx}")
            b[here] -= 2 * np.pi
            b[n_idx] += 2 * np.pi

            i = (i - 1) % len(neighbors_idx)

        prev_pos = here
        here = next_here
        maximum = -1

    # solve the linear system again to find the longitudes

    lons = np.linalg.solve(A, b).flatten()

    # add the boundary conditions for the north and south poles
    lons = np.append(lons, [np.nan, np.nan])

    longitudes = {idx_corners[i]: lons[i] for i in range(len(idx_corners))}

    return latitudes, longitudes


def optimize(corners, lats, longs, faces):
    import scipy.optimize

    idx_corner = list(corners.keys())
    corner_idx = {value: index for index, value in enumerate(idx_corner)}

    # build the variable vectors
    n = len(corners)

    x0 = np.zeros((n * 3), dtype=np.float64)
    for i in range(n):
        x0[3 * i : 3 * i + 3] = utils.azel_to_xyz(
            longs[idx_corner[i]], lats[idx_corner[i]]
        )

    # define the unit norm constraint
    def unit_norm_fn(x):
        # TODO could vectorize this by reshaping x maybe
        c = np.zeros(n, dtype=np.float64)

        for i in range(n):
            c[i] = np.linalg.norm(x[3 * i : 3 * i + 3])
        return c

    unit_norm = scipy.optimize.NonlinearConstraint(unit_norm_fn, 1, 1)

    def area_preservation_fcn(x):
        C = np.zeros((len(faces),), dtype=np.float64)
        for i, face in enumerate(faces):
            a, b, c, d = (
                corner_idx[face[0]],
                corner_idx[face[1]],
                corner_idx[face[2]],
                corner_idx[face[3]],
            )
            C[i] = utils.spherical_quadrilateral_area(
                x[3 * a : 3 * a + 3],
                x[3 * b : 3 * b + 3],
                x[3 * c : 3 * c + 3],
                x[3 * d : 3 * d + 3],
            )

        return C

    area_preservation = scipy.optimize.NonlinearConstraint(
        area_preservation_fcn, 4 * np.pi / len(faces), 4 * np.pi / len(faces)
    )

    # define the cost function
    def cost_fcn(x):
        return (np.abs(x - x0)).sum()

    res = scipy.optimize.minimize(
        cost_fcn, x0, constraints=[unit_norm, area_preservation], method="trust-constr"
    )
    print(res)
    print(res.x.flatten() - x0.flatten())


corners, visible_faces = create_data_structure_from_dense(voxels_d)

# print("BRUTE FORCING THE CORNERS ORDERING, PLEASE REMOVE THIS")
# corners_ordered = [
#     (1, 1, 1),
#     (2, 1, 1),
#     (3, 1, 1),
#     (1, 2, 1),
#     (2, 2, 1),
#     (3, 2, 1),
#     (1, 1, 2),
#     (2, 1, 2),
#     (3, 1, 2),
#     (1, 2, 2),
#     (2, 2, 2),
#     (3, 2, 2),
# ]

# corners = {k: corners[k] for k in corners_ordered}

lats, longs = initial_parametrization(corners, visible_faces)

optimize(corners, lats, longs, visible_faces)

# utils.plot_corners(visible_faces, lats)

utils.plot_corners_sphere(corners, lats, longs)
