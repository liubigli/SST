from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
from scipy.sparse import find, csr_matrix
from scipy.sparse.csgraph import connected_components

from SST.utils import resize_graph, get_positive_degree_nodes, get_subgraph
from ._morphological_segmentation import quasi_flat_zones
from SST.streaming import streaming_spanning_tree


def find_stable_and_unstable_nodes(e, cc, node_map):
    """

    Parameters
    ----------
    e:  NxN csr_matrix
        Adjacent matrix of the unstable edges

    cc: ndarray
        Connected components in the segmentation

    node_map: dict
        Dictionary that allows to map nodes in e to nodes of the compressed graph

    Returns
    -------
    stable_cc_nodes:
        Array containing nodes that are stable

    unstable_cc_nodes:
        Array containing nodes that are unstable
    """
    [e_src, e_dst, _] = find(e)

    e_seed_nodes = np.unique(np.concatenate((e_src, e_dst)))

    unstable_seed_nodes = np.array([node_map[e_node] for e_node in e_seed_nodes])

    unstable_labels = np.unique(cc[unstable_seed_nodes])

    # getting all the nodes contained in those cc
    all_nodes = np.arange(cc.shape[0])
    # unstable cc
    unstable_cc_nodes = np.concatenate([all_nodes[cc == label] for label in unstable_labels])

    stable_cc_nodes = np.setdiff1d(all_nodes, unstable_cc_nodes)

    return stable_cc_nodes, unstable_cc_nodes


def get_residual_graph(g, unstable_nodes, shape):
    """
    Function that returns the graph made of residual connected components that are unstable at the current iteration
    and we have to process again in the future iterations

    Parameters
    ----------
    g: csr_matrix
        Current graph

    unstable_nodes: ndarray
        Nodes of the unstable connected components in graph g

    shape: tuple
        shape of the resulting residual graph

    Returns
    -------
    residual_graph: csr_matrix
        Adjacency matrix of the residual graph made by the unstable connected components
    """

    # residual graph, i.e. graph made of nodes for which we cannot assign a label yet
    src, dst, w = find(g[unstable_nodes, :][:, unstable_nodes])
    res_graph = csr_matrix((w, (unstable_nodes[src], unstable_nodes[dst])), shape=shape)

    return res_graph


def quasi_flat_zone_streaming(stream_generator, threshold, return_img=False):
    """

    :param stream_generator:
    :param threshold:
    :param return_img:
    :return:
    """
    # generator of minimum spanning tree streaming
    mst_generator = streaming_spanning_tree(stream_generator, return_img=True)
    # minimum value for labels
    min_val_label = 0
    # residual graph
    res_graph = None

    for n, (t, e, i) in enumerate(mst_generator):
        # Computing quasi_flat_zones of the current graph
        # We have to distinguish different cases that depends both on the presence or not of a residual graph that comes
        # from previous iteration and on the presence or not of unstable edges in the current minimum spanning tree.
        # So in total we have 4 cases
        #   1: residual graph is empty and e is empty
        #   2: residual graph is empty and e is not empty
        #   3: residual graph is not empty and e is empty (i.e. last iteration)
        #   4: residual graph is not empty and e is not empty (i.e. common iteration)

        if n == 0:  # first iteration
            if e is None:  # pathological case in which we have only one image in our stream
                # computing quasi flat labels
                qf_labels = quasi_flat_zones(t, threshold)

                stable_nodes = np.arange(t.shape[0])

            else:
                # in the first iteration we do not have to remap nodes
                qf_labels = quasi_flat_zones((t+e), threshold)

                # find stable and unstable connected components
                # stable_nodes, unstable_nodes = find_stable_and_unstable_nodes(e, qf_labels, np.arange(len(qf_labels)))
                stable_nodes, unstable_nodes = find_stable_and_unstable_nodes(e,
                                                                              qf_labels,
                                                                              {i: i for i in range(len(qf_labels))})

                # removing unstable connected components from qf_labels and keeping only the stable ones
                qf_labels = qf_labels[stable_nodes]

                res_graph = get_residual_graph((t+e), unstable_nodes, t.shape)

        else:  # common iteration
            # reshaping graph
            res_graph = resize_graph(res_graph, t.shape)

            if e is None:  # last iteration
                current_graph = t.maximum(res_graph)

                # collecting only nodes with a positive degree
                nodes = get_positive_degree_nodes(current_graph)

                # extracting subgraph containing info only on those nodes
                g, map_nodes = get_subgraph(current_graph, nodes, return_map=True)

                # computing quasi flat zones
                qf_labels = quasi_flat_zones(g, threshold) + min_val_label

                stable_nodes = nodes

            else:
                current_graph = t.maximum(e).maximum(res_graph)

                nodes = get_positive_degree_nodes(current_graph)

                g, map_nodes = get_subgraph(current_graph, nodes, return_map=True)

                # computing quasi flat zones
                qf_labels = quasi_flat_zones(g, threshold) + min_val_label

                # find stable and unstable connected components
                stable_nodes, unstable_nodes = find_stable_and_unstable_nodes(e, qf_labels, map_nodes)

                # removing unstable connected components from qf_labels and keeping only the stable ones
                qf_labels = qf_labels[stable_nodes]

                # remapping stable nodes
                stable_nodes = nodes[stable_nodes]

                res_graph = get_residual_graph(current_graph, nodes[unstable_nodes], t.shape)

        # updating min_val_label for the next iteration in order to not merge different connected components
        min_val_label = qf_labels.max() + 1

        # fetching array of segmented nodes
        segmentation = np.vstack((stable_nodes, qf_labels))

        if return_img:
            yield segmentation, i
        else:
            yield segmentation
