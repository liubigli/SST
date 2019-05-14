from scipy.sparse.csgraph import minimum_spanning_tree
from SST.streaming.graph import get_half_cycle
from SST.utils import resize_graph, merge_graphs

def streaming_spanning_tree_v1(streaming_generator, return_img=False):
    r"""
    Function that implement our first streaming version of minimum spanning tree

    Parameters
    ----------
    streaming_generator: generator
        Generator of the streaming
        At each iteration it should yield:
            - the ith block of the image
            - the graph associated to the ith block of image
            - the root id of the graph
            - the ids of the vertices in the common border with the graph in the next iteration

    return_img: bool (default False)
        If true the method yields also ith image int the generator otherwise no.
    """
    old_e = None
    old_root = None
    old_front = None
    for img, graph, root, front in streaming_generator:
        t = minimum_spanning_tree(graph)

        if old_e is None:  # first iteration
            old_e = get_half_cycle(t, root, front)
            t = t - old_e
            old_root = root
            old_front = front

            if return_img:
                yield t, old_e, img
            else:
                yield t, old_e

        else:
            # we assume that the size of the graph increases during the iterations
            old_e = resize_graph(old_e, graph.shape)
            t, e = _update_minimum_spanning_tree(t, old_e, old_front, old_root, front, root)

            # updating variables for next iteration
            old_e = e
            old_front = front
            old_root = root

            if return_img:
                yield t, old_e, img
            else:
                yield t, e


def _update_minimum_spanning_tree(t_new, e_old, front_old, root_old, front_new=None, root_new=-1):
    """Function that compute the nth iteration of minimum spanning tree algorithm for images
    Parameters
    ----------
    t_new: csr_matrix
        MST of the newly arrived graph
    e_old: csr_matrix
        Edges in the old tree that are supposed to form cycles

    front_old: ndarray
        Frontier of the newly arrived graph with the old graph

    root_old: int
        id of the node used as root in the previous iteration

    front_new: ndarray
        Frontier of the newly arrived graph with further graph

    root_new: int
        ID of the node that we are going to use as root in this iteration

    """
    # find edges that form cycles in the fusion between the fusion of two mst
    e_new = get_half_cycle(t_new, root_old, front_old)

    # removing from new_t the edges that form cycles
    t_new = t_new - e_new

    # the cycles the merged graph are defined by the maximum between e_old and e_new
    e_in_cycles = merge_graphs([e_old, e_new])

    # computing the mst we remove all the cycles in e_in_cycles
    mst_e_in_cycles = minimum_spanning_tree(e_in_cycles)

    t_n = t_new + mst_e_in_cycles

    if front_new is not None:
        # computing edges in cycles for the next iteration
        e_n = get_half_cycle(t_n, root_new, front_new)

        return t_n-e_n, e_n
    # we always return two elements
    return t_n, None


def streaming_spanning_tree_v2(streaming_generator, return_img=False):
    """
    Function that implement our streaming version of minimum spanning tree

    Parameters
    ----------
    streaming_generator: generator
        Generator of the streaming.
        At each iteration it should yield:
            - the ith block of image
            - the graph associated to the ith block of image
            - the root id of the graph
            - the ids of the vertices in the common border with the graph in the next iteration

    return_img: bool
        If true the algorithm yields also ith image in the generator otherwise no. Default is False
    """
    e = None

    for img, graph, root, front in streaming_generator:

        if e is None:
            t = minimum_spanning_tree(graph)
        else:
            # we assume that the size of the graph increases during the iterations
            e = resize_graph(e, graph.shape)
            graph = graph.multiply(graph > 0).maximum(e.multiply(e > 0)) + \
                    graph.multiply(graph < 0).minimum(e.multiply(e < 0))
            t = minimum_spanning_tree(graph)

        if front is not None:
            e = get_half_cycle(t, root, front)
            t = t - e
        else:
            e = None

        if return_img:
            yield t, e, img

        else:
            yield t, e
