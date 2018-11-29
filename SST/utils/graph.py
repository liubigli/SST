from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.sparse import find, csr_matrix


def resize_graph(graph, gshape, labels=None):
    """
    Change size of a graph

    Parameters
    ----------
    graph: csr_matrix
        Graph to resize

    gshape: tuple
        New size of the resized graph

    labels: ndarray
        Optional remapping to apply to nodes of the initial graph

    Returns
    -------
    graph: csr_matrix
        Representing the graph with new shape

    """
    [src, dest, weights] = find(graph)

    if labels is None:
        return csr_matrix((weights, (src, dest)), shape=gshape)

    return csr_matrix((weights, (labels[src], labels[dest])), shape=gshape)


def shuffle_labels(labels):
    """
    Function that shuffle labels of a segmentation

    Parameters
    ----------
    labels: N ndarray
        input labels

    Returns
    -------
    labels: N ndarray
        vector of shuffled labels

    """
    unique = np.unique(labels)
    np.random.seed(1)
    np.random.shuffle(unique)

    return unique[labels]


# auxiliary function that reconstruct the ith mst from the sequences T,E
def reconstruct_ith_mst(T, E, ith=0):
    # easy case
    if ith == 0:
        return T[0] + E[0]

    if ith > len(T) - 1 or ith < 0:
        ith = len(T) - 1

    max_shape = max(t.shape for t in T[:ith + 1])

    # resizing all the graphs in the lists
    for i in range(ith + 1):
        T[i] = resize_graph(T[i], max_shape)

    mst = sum(T[:ith + 1])

    if E[ith] is not None:
        mst += E[ith]

    return mst
