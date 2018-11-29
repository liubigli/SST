from scipy.sparse.csgraph import minimum_spanning_tree
from SST.streaming.graph import get_half_cycle
from SST.utils import resize_graph


def streaming_spanning_tree(streaming_generator):
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

    """
    e = None

    for img, graph, root, front in streaming_generator:

        if e is None:
            t = minimum_spanning_tree(graph)
        else:
            e = resize_graph(e, graph.shape)
            t = minimum_spanning_tree(graph.maximum(e))

        if front is not None:
            e = get_half_cycle(t, root, front)
            t = t - e
        else:
            e = None

        yield img, t, e