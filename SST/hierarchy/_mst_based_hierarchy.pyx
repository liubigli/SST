from __future__ import absolute_import

import numpy as np
cimport numpy as np
cimport cython
import time
from scipy.sparse.csgraph._validation import validate_graph
from scipy.sparse import csr_matrix


"""PARAMETERS.pxi"""
include 'parameters.pxi'

# Auxiliary variables for the alpha_omega_segmentation
COL_V1 = 0
COL_V2 = 1
COL_R1 = 2
COL_R2 = 3
COL_CHILD_LEFT = 4
COL_CHILD_RIGHT = 5
COL_W = 6
COL_RNG = 7
COL_MIN1 = 8
COL_MIN2 = 9
COL_MAX1 = 10
COL_MAX2 = 11
NUM_COL = 12


def hierarchy_from_mst(csgraph, vertex_values=None, overwrite=False):
    """
    Parameters
    ----------
    csgraph: csr_matrix
        minimum spanning tree
    vertex_values: ndarray
        values in the image
    overwrite: boolean

    Returns
    -------
    hierarchy_tree: ndarray

    """

    global NULL_IDX

    csgraph = validate_graph(csgraph, True, DTYPE, dense_output=False, copy_if_sparse=not overwrite)

    # number of nodes in minimum spanning tree

    N = csgraph.shape[0]

    data = csgraph.data
    indices = csgraph.indices
    indptr = csgraph.indptr

    rank = np.zeros(N, dtype=ITYPE)
    predecessors = np.arange(N, dtype=ITYPE)

    i_sort = np.argsort(data).astype(ITYPE)
    row_indices = np.zeros(len(data), dtype=ITYPE)

    vertex_values = vertex_values.astype(DTYPE)

    if len(vertex_values.shape) > 1:
        hierarchy_tree = -np.ones((N - 1, COL_RNG + 1 + 4*vertex_values.shape[1]), dtype=DTYPE)
        ranges = np.repeat(vertex_values, 2).reshape(vertex_values.shape[0], 2 * vertex_values.shape[1])
    else:
        hierarchy_tree = -np.ones((N - 1, NUM_COL), dtype=DTYPE)
        ranges = np.repeat(vertex_values, 2).reshape(vertex_values.shape[0], 2)
    # information on childs
    history = -np.ones(N, dtype=ITYPE)

    _build_hierarchy(hierarchy_tree,
                     ranges,
                     history,
                     data,
                     indices,
                     indptr,
                     i_sort,
                     row_indices,
                     predecessors,
                     rank)

    return hierarchy_tree


@cython.boundscheck(False)
@cython.wraparound(False)
def _build_hierarchy(DTYPE_t[:,::1] hierarchy_tree,
                     DTYPE_t[:,::1] ranges,
                     ITYPE_t[::1] history,
                     DTYPE_t[::1] data,
                     ITYPE_t[::1] col_indices,
                     ITYPE_t[::1] indptr,
                     ITYPE_t[::1] i_sort,
                     ITYPE_t[::1] row_indices,
                     ITYPE_t[::1] predecessors,
                     ITYPE_t[::1] rank):


    cdef unsigned int i, j, V1, V2, R1, R2, n_edges_in_mst, n_verts, n_data, c
    cdef int child_left, child_right
    cdef DTYPE_t W, rng
    channels = range(int(ranges.shape[1] / 2))

    n_verts = predecessors.shape[0]
    n_data = i_sort.shape[0]

    # Arrange `row_indices` to contain the row index of each value in `data`.
    # Note that the array `col_indices` already contains the column index.
    for i in range(n_verts):
        for j in range(indptr[i], indptr[i + 1]):
            row_indices[j] = i
    # step through the edges from smallest to largest.
    #  V1 and V2 are connected vertices.
    n_edges_in_mst = 0
    i = 0
    while i < n_data and n_edges_in_mst < n_verts - 1:
        j = i_sort[i]
        V1 = row_indices[j]
        V2 = col_indices[j]
        W = data[j]

        # progress upward to the head node of each subtree
        R1 = V1
        while predecessors[R1] != R1:
            R1 = predecessors[R1]
        R2 = V2
        while predecessors[R2] != R2:
            R2 = predecessors[R2]

        # Compress both paths.
        while predecessors[V1] != R1:
            predecessors[V1] = R1
        while predecessors[V2] != R2:
            predecessors[V2] = R2

        # if the subtrees are different, then we connect them and keep the
        # edge.  Otherwise, we remove the edge: it duplicates one already
        # in the spanning tree.
        if R1 != R2:

            """<---------- BEGIN UPTDATE TABLE ---------->"""
            hierarchy_tree[n_edges_in_mst, COL_V1] = V1
            hierarchy_tree[n_edges_in_mst, COL_V2] = V2
            hierarchy_tree[n_edges_in_mst, COL_R1] = R1
            hierarchy_tree[n_edges_in_mst, COL_R2] = R2
            hierarchy_tree[n_edges_in_mst, COL_CHILD_LEFT] = history[R1]
            hierarchy_tree[n_edges_in_mst, COL_CHILD_RIGHT] = history[R2]
            hierarchy_tree[n_edges_in_mst, COL_W] = W
            rng = 0


            for c in channels:
                hierarchy_tree[n_edges_in_mst, COL_MIN1 + 4 * c] = ranges[R1, 2 * c]
                hierarchy_tree[n_edges_in_mst, COL_MIN2 + 4 * c] = ranges[R2, 2 * c]
                hierarchy_tree[n_edges_in_mst, COL_MAX1 + 4 * c] = ranges[R1, 2 * c + 1]
                hierarchy_tree[n_edges_in_mst, COL_MAX2 + 4 * c] = ranges[R1, 2 * c + 1]
                # REMARK THAT WE ARE USING AS RANGE THE MAX RANGE AMONG ALL THE CHANNELS
                rng = max(rng, np.abs(
                    max(ranges[R1, 2 * c + 1], ranges[R2, 2 * c + 1]) - min(ranges[R1, 2 * c], ranges[R2, 2 * c])))

            hierarchy_tree[n_edges_in_mst, COL_RNG] = rng


            """<---------- END UPTDATE TABLE ---------->"""

            n_edges_in_mst += 1

            # Use approximate (because of path-compression) rank to try
            # to keep balanced trees.
            if rank[R1] > rank[R2]:
                predecessors[R2] = R1
                history[R1] = n_edges_in_mst - 1
                for c in channels:
                    ranges[R1, 2 * c] = min(ranges[R1, 2 * c], ranges[R2, 2 * c])
                    ranges[R1, 2 * c + 1] = max(ranges[R1, 2 * c + 1], ranges[R2, 2 * c + 1])

            elif rank[R1] < rank[R2]:

                predecessors[R1] = R2
                history[R2] = n_edges_in_mst - 1
                for c in channels:
                    ranges[R2, 2 * c] = min(ranges[R1, 2 * c], ranges[R2, 2 * c])
                    ranges[R2, 2 * c + 1] = max(ranges[R1, 2 * c + 1], ranges[R2, 2 * c + 1])
            else:
                predecessors[R2] = R1
                history[R1] = n_edges_in_mst - 1
                rank[R1] += 1
                for c in channels:
                    ranges[R1, 2 * c] = min(ranges[R1, 2 * c], ranges[R2, 2 * c])
                    ranges[R1, 2 * c + 1] = max(ranges[R1, 2 * c + 1], ranges[R2, 2 * c + 1])

        else:
            data[j] = 0

        i += 1


    # propagating range attribute from top to down
    for i in range(n_verts-2, -1, -1):
        child_right = int(hierarchy_tree[i, COL_CHILD_RIGHT])

        if child_right >= 0 and hierarchy_tree[i, COL_W] == hierarchy_tree[child_right, COL_W]:
            hierarchy_tree[child_right, COL_RNG] = hierarchy_tree[i, COL_RNG]

        child_left = int(hierarchy_tree[i, COL_CHILD_LEFT])

        if child_left >= 0 and hierarchy_tree[i, COL_W] == hierarchy_tree[child_left, COL_W]:

            hierarchy_tree[child_left, COL_RNG] = hierarchy_tree[i, COL_RNG]
