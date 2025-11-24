import numpy as np


def sparse_to_dense(voxels):
    # find the minimun and maximum grid index

    # assume that the minimum is [0,0,0] to be able to go back and forth
    gi_max = np.array([-1, -1, -1])

    for voxel in voxels:
        if voxel[0] > gi_max[0]:
            gi_max[0] = voxel[0]
        if voxel[1] > gi_max[1]:
            gi_max[1] = voxel[1]
        if voxel[2] > gi_max[2]:
            gi_max[2] = voxel[2]

    voxels_dense = np.full(
        1 + gi_max + 2,  # add one row above and below
        False,
    )

    for voxel in voxels:
        idx = np.array(voxel) + np.array([1, 1, 1])  # compensate for the padding
        voxels_dense[tuple(idx)] = True

    return voxels_dense


def dense_to_sparse(voxels):
    return np.argwhere(voxels) - 1  # compensate for the padding


if __name__ == "__main__":
    voxels = [
        tuple([0, 0, 0]),
        tuple([0, 0, 1]),
        tuple([0, 1, 0]),
        tuple([0, 1, 1]),
        tuple([1, 0, 0]),
        tuple([1, 0, 1]),
        tuple([1, 1, 0]),
        tuple([1, 1, 1]),
    ]

    dense = sparse_to_dense(voxels)
    sparse = dense_to_sparse(dense)


def plot_corners(visible_faces, colors):

    import pyvista as pv
    import numpy as np

    # Initialize lists
    points = []
    faces = []
    scalars = []

    corner_set = set()

    def add_square(corners, corner_keys, scalar_values):
        """
        Add a square to the mesh.

        Parameters:
        - corners: array of shape (4, 3)
        - corner_keys: list of 4 keys (e.g. corner IDs)
        - scalar_values: list of 4 scalar values
        """
        valids = [v for v in scalar_values if not np.isnan(v)]
        mean = np.mean(valids) if valids else 0.0
        scalar_values = [v if not np.isnan(v) else mean for v in scalar_values]

        idx = len(points)

        points.extend(corners)
        scalars.extend(scalar_values)
        for corner in corner_keys:
            corner_set.add(tuple(int(c) for c in corner))

        faces.extend([4, idx, idx + 1, idx + 2, idx + 3])

    for face in visible_faces:
        corner_keys = face
        add_square(
            corners=[list(c) for c in face],
            corner_keys=corner_keys,
            scalar_values=[colors[corner] for corner in face],
        )

    # Convert lists to arrays
    points_np = np.array(points)
    faces_np = np.array(faces)
    scalars_np = np.array(scalars)

    # Create mesh
    mesh = pv.PolyData(points_np, faces_np)
    mesh.point_data["scalars"] = scalars_np

    # Plot
    plotter = pv.Plotter()
    plotter.add_mesh(
        mesh,
        scalars="scalars",
        cmap="viridis",
        show_edges=True,
        interpolate_before_map=True,
    )

    plotter.add_point_labels(
        list(corner_set),
        [str(c) for c in corner_set],
        font_size=12,
        text_color="black",
        point_size=10,
        shape_opacity=0.3,
        always_visible=False,
    )
    plotter.show()


def plot_corners_sphere(corners, lats, longs):

    import pyvista as pv
    import numpy as np

    mapping = {}

    # Initialize point list and face list
    plotter = pv.Plotter()
    plotter.add_mesh(
        pv.Sphere(radius=1, theta_resolution=60, phi_resolution=60),
        opacity=1,
        color="white",
    )

    lines = set()
    for corner in corners.keys():
        neighbors = get_neighbors(corners, corner, direct=True, sort=False)

        a_sphere = azel_to_xyz(longs[corner], lats[corner])

        mapping[tuple([int(c) for c in corner])] = a_sphere

        for neighbor in neighbors:
            line = tuple(sorted((corner, neighbor)))

            if line not in lines:
                lines.add(line)
                # project on the sphere
                b_sphere = azel_to_xyz(longs[neighbor], lats[neighbor])

                ts = np.linspace(0, 1, 100)
                arc_points = slerp(a_sphere, b_sphere, ts)
                plotter.add_mesh(
                    pv.Spline(arc_points, 100), color="black", line_width=3
                )

    # Plot the mesh with interpolated scalar coloring

    plotter.add_point_labels(
        list(mapping.values()),
        list(mapping.keys()),
        font_size=12,
        text_color="black",
        point_size=10,
        shape_opacity=0.3,
        always_visible=False,
    )
    plotter.show()


def azel_to_xyz(az, el):
    if np.isnan(az):
        az = 0.0
    return np.sin(el) * np.cos(az), np.sin(el) * np.sin(az), np.cos(el)


def xyz_to_azel(x, y, z):
    az = np.arctan2(y, x) % (2 * np.pi)
    el = np.arccos(z)
    return az, el


def slerp(a, b, t_array):
    """Spherical linear interpolation between points a and b"""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    dot = np.clip(np.dot(a, b), -1.0, 1.0)
    omega = np.arccos(dot)
    sin_omega = np.sin(omega)

    if sin_omega < 1e-6:
        # Points are too close or opposite; fall back to linear interpolation
        return np.array([(1 - t) * a + t * b for t in t_array])

    return np.array(
        [
            (np.sin((1 - t) * omega) * a + np.sin(t * omega) * b) / sin_omega
            for t in t_array
        ]
    )


def angle_between(u, v, w):
    # Compute the spherical angle at vertex v between vectors u->v and w->v
    a = np.cross(u, v)
    b = np.cross(w, v)
    numerator = np.dot(a, b)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)  # should be 1 already
    return np.arccos(np.clip(numerator / denominator, -1.0, 1.0))


def spherical_triangle_area(a, b, c, radius=1.0):
    A = angle_between(c, a, b)
    B = angle_between(a, b, c)
    C = angle_between(b, c, a)
    E = A + B + C - np.pi
    return E * radius**2


def spherical_quadrilateral_area(A, B, C, D, radius=1.0):
    # Divide into triangles ABC and CDA
    area1 = spherical_triangle_area(A, B, C, radius)
    area2 = spherical_triangle_area(C, D, A, radius)
    return area1 + area2


def get_neighbors(corners, corner: np.ndarray, direct=True, sort=True):
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
