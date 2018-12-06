from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
from scipy.sparse import find, csr_matrix
from scipy.sparse.csgraph import connected_components

from SST.utils import resize_graph, get_positive_degree_nodes, get_subgraph
from ._morphological_segmentation import quasi_flat_zones
from SST.streaming import streaming_spanning_tree


def map_nodes_to_sub_graph(graph, node_map):
    """
    Function that maps all the nodes with a positive degree in given graph to a subgraph.
    The mapping is defined by the dictionary node_map.

    Parameters
    ----------
    graph: csr_matrix
        Adjacent matrix of graph

    node_map: dict
        Dictionary containing information for the mapping

    Returns
    -------
    remapped_nodes: M ndarray
        Array of the nodes in the subgraph
    """

    # retrieving all the nodes in the graph with a positive degree
    nodes = get_positive_degree_nodes(graph)

    # array of remapped nodes
    remapped_nodes = np.array([node_map[node] for node in nodes])

    return remapped_nodes


def find_stable_and_unstable_nodes(cc, unstable_seed_nodes):
    """
    Method that returns two arrays. The first is made of nodes belongings to stable connected components, the second
    is made of nodes belongings to unstable connected components.

    Parameters
    ----------
    cc: ndarray
        Connected components in the segmentation

    unstable_seed_nodes: ndarray
        Array containing nodes in the unstable part of the minimum spanning tree. Those nodes are seed to retrieve
        all the unstable connected components.

    Returns
    -------
    stable_cc_nodes:
        Array containing nodes that are stable

    unstable_cc_nodes:
        Array containing nodes that are unstable

    """

    # array of labels that are unstable at the current iteration.
    unstable_labels = np.unique(cc[unstable_seed_nodes])

    # getting all the nodes contained in those cc
    all_nodes = np.arange(cc.shape[0])

    # nodes in the unstable connected components
    unstable_cc_nodes = np.concatenate([all_nodes[cc == label] for label in unstable_labels])

    # nodes in the stable connected components
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
    Parameters
    ----------
    stream_generator: generator
        Generator of the streaming.
        At each iteration it should yield:
            - the ith block of image
            - the graph associated to the ith block of image
            - the root id of the graph
            - the ids of the vertices in the common border with the graph in the next iteration

    threshold: float
        Value used to remove edges of the minimum spanning tree. All the edges whose weight is above threshold value
        are removed.

    return_img: bool
        If true at each iteration the function yields also the current image in the streaming

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

                # get nodes with positive degree in e
                remapped_e_nodes = get_positive_degree_nodes(e)

                # find stable and unstable connected components
                stable_nodes, unstable_nodes = find_stable_and_unstable_nodes(qf_labels, remapped_e_nodes)

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

                # compressing current graph in order to deal with less information in memory
                g, map_nodes = get_subgraph(current_graph, nodes, return_map=True)

                # computing quasi flat zones
                qf_labels = quasi_flat_zones(g, threshold) + min_val_label

                # remapping nodes in graph e to nodes in sub graph g
                remapped_e_nodes = map_nodes_to_sub_graph(e, map_nodes)

                # find stable and unstable connected components
                # stable_nodes, unstable_nodes = find_stable_and_unstable_nodes(e, qf_labels, map_nodes)
                stable_nodes, unstable_nodes = find_stable_and_unstable_nodes(qf_labels, remapped_e_nodes)

                # removing unstable connected components from qf_labels and keeping only the stable ones
                qf_labels = qf_labels[stable_nodes]

                # remapping stable nodes to the initial graph
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


def get_all_nodes_but_well(g, id_well):
    """
    Function that return all nodes with a positive degree in graph except for the well node.

    Parameters
    ----------
    g: csr_matrix
        Adjacent matrix of the graph g
    id_well: int
        Id of the well node

    Returns
    -------
    nodes: ndarray
        list of nodes in g
    """
    # collecting all the nodes
    nodes = get_positive_degree_nodes(g)

    # returning all the ids except for the well
    return np.setdiff1d(nodes, id_well)


def marker_flooding_streaming(stream_generator, return_img=False):
    """
    Streaming version of the watershed algorithm on graph.
    See as reference Jean Cousty et Al. ( https://hal.inria.fr/hal-01113462/document )
    This function is analogous to quasi_flat_streaming function

    Parameters
    ---------
    stream_generator:
        Generator for streaming image and markers.
        At each iteration it should yield:
            - the ith block of image
            - the graph associated to the ith block of image
            - the root id of the graph
            - the ids of the vertices in the common border with the graph in the next iteration

    return_img: bool
        If true at each iteration the function yields also the current image in the streaming
    """
    # TODO: probably this code can be merged with the one in quasi_flat_zone_streaming
    mst_generator = streaming_spanning_tree(stream_generator, return_img=True)

    # residual graph
    res_graph = None

    # minimum value for labels
    min_val_label = 1
    for n, (t, e, i) in enumerate(mst_generator):
        if n == 0:
            # we suppose that the well node is the last node
            id_ext_node = t.shape[0] - 1

            if e is None:  # patological case in which the stream is made by only one iteration
                # Remark in this case id_ext_node = t.shape[0] - 1
                ncc, labels = connected_components(t[:id_ext_node, :id_ext_node], directed=False)
                stable_nodes = np.arange(len(labels))
            else:

                e_nodes = get_all_nodes_but_well(e, id_ext_node)
                t_nodes = get_all_nodes_but_well(t+e, id_ext_node)

                # taking all the nodes except id_ext_node
                nodes = np.unique(np.concatenate((e_nodes, t_nodes)))

                g = get_subgraph(t+e, nodes)

                _, labels = connected_components(g, directed=False)

                stable_nodes, unstable_nodes = find_stable_and_unstable_nodes(labels, e_nodes)

                labels = labels[stable_nodes]

                res_graph = get_residual_graph((t+e), unstable_nodes, t.shape)
                # update res_graph and yield connected components

        else:
            if e is None:  # last iteration
                current_graph = t.maximum(res_graph)

                # collecting nodes with a positive degree
                nodes = get_all_nodes_but_well(current_graph, id_well=id_ext_node)

                # extracting subgraph containing info only on those nodes
                g, map_nodes = get_subgraph(current_graph, nodes, return_map=True)

                # computing connected components
                ncc, labels = connected_components(g, directed=False)
                labels += min_val_label

                stable_nodes = nodes

            else:
                current_graph = t.maximum(e).maximum(res_graph)

                nodes = get_all_nodes_but_well(current_graph, id_ext_node)
                e_nodes = get_all_nodes_but_well(e, id_ext_node)
                # compressing current graph in order to deal with less information in memory
                g, map_nodes = get_subgraph(current_graph, nodes, return_map=True)

                # computing quasi flat zones
                ncc, labels = connected_components(g, directed=False)
                labels += min_val_label

                # remapping nodes in graph e to nodes in sub graph g
                remapped_e_nodes = np.array([map_nodes[e_node] for e_node in e_nodes])

                # find stable and unstable connected components
                # stable_nodes, unstable_nodes = find_stable_and_unstable_nodes(e, qf_labels, map_nodes)
                stable_nodes, unstable_nodes = find_stable_and_unstable_nodes(labels, remapped_e_nodes)

                # removing unstable connected components from qf_labels and keeping only the stable ones
                labels = labels[stable_nodes]

                # remapping stable nodes to the initial graph
                stable_nodes = nodes[stable_nodes]

                res_graph = get_residual_graph(current_graph, nodes[unstable_nodes], t.shape)

        min_val_label = labels.max() + 1

        segmentation = np.vstack((stable_nodes, labels))

        if return_img:
            yield segmentation, i
        else:
            yield segmentation
