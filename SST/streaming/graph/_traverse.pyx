from __future__ import absolute_import

import numpy as np

cimport numpy as np
cimport cython

from scipy.sparse.csgraph._validation import validate_graph
from scipy.sparse import csr_matrix

from scipy.sparse.csgraph import depth_first_order

include 'parameters.pxi'


def traverse_up_to_root(tree, vertices, predecessors, overwrite=False):
    """
    Function that returns the graph of all the paths from side_vertices to the root of the tree

    Parameters
    ----------
    tree: NxN csr_matrix
        sparse matrix, 2 dimension  representing adjacency matrix of the tree

    vertices: ndarray
        list of vertices for which we compute the paths

    predecessors: ndarray
        array of tree predecessors

    overwrite: bool
        optional f true, then parts of the input graph will be overwritten for efficiency.

    Returns
    -------
    g: csr_matrix
        The NxN compressed-sparse representation of the undirected paths from list of vertices to root

    """
    tree = validate_graph(tree, True, DTYPE, dense_output=False, copy_if_sparse=not overwrite)
    # number of vertices
    N = predecessors.shape[0]
    visited = np.zeros(N, dtype=ITYPE)
    data = tree.data
    indices = tree.indices
    indptr = tree.indptr
    row_indices = np.zeros(len(data), dtype=ITYPE)
    vertices = vertices.astype(ITYPE)

    _traverse_up_to_root(data, indices, indptr, row_indices, predecessors, visited, vertices)

    g = csr_matrix((data, indices, tree.indptr), tree.shape)
    g.eliminate_zeros()

    return g


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _traverse_up_to_root(DTYPE_t[::1] data,
                                ITYPE_t[::1] col_indices,
                                ITYPE_t[::1] indptr,
                                ITYPE_t[::1] row_indices,
                                ITYPE_t[::1] predecessors,
                                ITYPE_t[::1] visited,
                                ITYPE_t[::1] side_vertices) nogil:

    cdef unsigned int i, j, k, node, n_verts, n_side_verts, n_ind
    n_side_verts = side_vertices.shape[0]
    n_verts = predecessors.shape[0]
    n_ind = col_indices.shape[0]

    # Arrange `row_indices` to contain the row index of each value in `data`.
    # Note that the array `col_indices` already contains the column index.
    for i in range(n_verts):
        for j in range(indptr[i], indptr[i + 1]):
            row_indices[j] = i

    for i in range(n_side_verts):
        node = side_vertices[i]

        while predecessors[node] >= 0 and visited[node] == 0:
            # update the visited vector
            visited[node] = 1
            # next vertex that we want to visit is the parent of node in the tree
            node = predecessors[node]

    # add visited equals to 1 for the source of the tree
    visited[side_vertices[0]] = 1

    for i in range(n_ind):
        j = col_indices[i]
        k = row_indices[i]
        if visited[j] == 0 or visited[k] == 0:
            data[i] = 0


def get_side_vertices(shape, side='left'):
    """
    Returns the list of indices for nodes in a side of an image of a given shape

    Parameters
    ----------
    shape: tuple
        shape of the image

    side: {'left', 'right', 'top', 'bottom'} optional
        Side of the image for which we compute the indices.

    vertices: ndarray
        array of indices of pixels on the given side of an image with dimension equals to shape
    """
    if side=='left':
        return shape[1]*np.arange(shape[0], dtype=ITYPE)
    if side=='right':
        return shape[1]*np.arange(shape[0], dtype=ITYPE) + (shape[1] - 1)
    if side=='top':
        return np.arange(shape[1], dtype=ITYPE)
    if side=='bottom':
        return np.arange(shape[1], dtype=ITYPE) + (shape[0] - 1)*shape[1]

    raise KeyError("No value for side %s. Only values admitted are left, top, right, bottom" % side)


def edges_in_cycles(shape, tree, side='left'):
    """
    Auxiliary function that

    Parameters
    ----------
    shape: tuple
        Shape of the image

    tree: NxN csr_matrix
        Sparse matrix, 2 dimensions. Adjacency matrix of the tree.

    side: {'right', 'bottom'} optional
        Side of the image for which find the paths to root of the tree.

    Returns
    -------
    graph: NxN csr_matrix
        The NxN compressed-sparse representation of the undirected paths from list of vertices in side

    """
    nr, nc = shape[0:2]
    side_vertices = get_side_vertices(shape, side)
    src_pixel = 0
    if side=='right':
        src_pixel = nc-1
    if side=='bottom':
        src_pixel = (nr-1)*nc

    _, predecessors = depth_first_order(tree, src_pixel, directed=False, return_predecessors=True)

    return traverse_up_to_root(tree, side_vertices, predecessors)


def get_half_cycle(tree, root, vertices):
    """
    Function that given a minimum spanning tree returns the edges that are supposed to form cycle after merging
    with the new graph

    Parameters
    ----------
    tree: NxN csr_matrix
        current minumum spanning tree

    root: int
        id of the root for the tree

    vertices: ndarray
        Array with ids of nodes in the common frontier between the current tree and the new graph

    Returns
    -------
    graph: csr_matrix
        Adjacent matrix of the graph containing edges that are supposed to form cycles with the new graph

    """

    _, predecessors = depth_first_order(tree, root, directed=False, return_predecessors=True)

    return traverse_up_to_root(tree, vertices, predecessors)