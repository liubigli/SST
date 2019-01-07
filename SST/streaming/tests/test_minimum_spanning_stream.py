import numpy as np
from scipy.misc import ascent
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components

from SST.utils import img_to_graph, stick_two_images, resize_graph
from SST.streaming.streaming_generators import HorizontalStreaming
from SST.streaming import streaming_spanning_tree


def test_minimum_spanning_stream():

    # loading test image
    im = ascent()
    img = im[250:350]

    # initializing streaming generator
    gen = HorizontalStreaming(img)
    stream = gen.generate_stream(block_shape=(100, 100))

    curr_img = None
    stable_graph = None

    for n, (t, e, i) in enumerate(streaming_spanning_tree(stream, return_img=True)):
        if n == 0:
            curr_img = i
            stable_graph = t
        else:
            curr_img = stick_two_images(curr_img, i, num_overlapping=1, direction='H')
            stable_graph = resize_graph(stable_graph, t.shape)
            stable_graph += t

        # extracting real minimum spanning tree
        real_mst = minimum_spanning_tree(img_to_graph(curr_img))

        if e is not None:
            mst = stable_graph + e
        else:  # last iteration
            mst = stable_graph

        # testing that the mst found using our algorithm is always connected
        ncc, _ = connected_components(mst)
        assert ncc == 1

        # asserting that the mst found is a tree (=> thus spans all nodes)
        assert mst.nnz == (mst.shape[0] - 1)

        # asserting that mst is minimal
        assert np.around(mst.sum(), decimals=9) == np.around(real_mst.sum(), decimals=9)
