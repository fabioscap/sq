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
