from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABC, abstractmethod
import numpy as np
from scipy.sparse import csr_matrix, find

from SST.utils import img_to_graph, resize_graph

class AbstractImageStreamingGenerator(ABC):
    """
    Abstract class that we will use create several generator starting from images
    """
    def __init__(self, img):
        self.img = np.atleast_3d(img)
        super(AbstractImageStreamingGenerator, self).__init__()

    @abstractmethod
    def generate_stream(self, block_shape, markers=None, return_map=False):
        """
        Function that generates a stream of images all of the same dimension.

        Parameters
        ----------
        block_shape: tuple
            Size of the block image

        markers: list
            List of markers in the image

        return_map: bool (default False)
            If true the method should yields also an additional image that represent a map between image pixels and
            nodes of the graph associated to the image.
        """
        pass


class HorizontalStreaming(AbstractImageStreamingGenerator):
    def generate_stream(self, block_shape, markers=None, return_map=False):
        """
        Function that generate a Horizontal stream of image all with the same size.

        Parameters
        ----------
        block_shape: tuple
            Size of the block image

        markers: list
            List of markers in the image

        return_map: bool (default False)
            If true the method should yields also an additional image that represent a map between image pixels and
            nodes of the graph associated to the image.

        """
        # image size
        nr, nc, nz = self.img.shape

        # size of streaming block
        b_nr, b_nc = block_shape

        # we compute the number of iterations
        max_it = np.ceil((nc - b_nc) / (b_nc - 1)).astype(np.int) + 1

        # initializing min_col for the first iteration
        ith_min_col = 0

        id_ext_node = nr*nc

        for i in range(max_it):
            # max col of the ith img
            ith_max_col = min((b_nc * (i+1)) - i, nc)

            # getting ith block
            img = self.img[:b_nr, ith_min_col:ith_max_col]

            # case for the grayscale image
            if nz == 1:
                img = img[:, :, 0]

            # getting graph associated to ith image block
            g = img_to_graph(img, order='F', col_offset=ith_min_col)

            # id root node for the ith tree
            root = b_nr * (ith_max_col - 1) if i < max_it - 1 else None

            # id nodes front for the ith iteration
            front_nodes = np.arange(b_nr * (ith_max_col - 1), b_nr * ith_max_col, dtype=int) if i < max_it - 1 else None

            if markers is not None:  # case segmentation with markers
                # fetching markers in current image block
                markers_in_block = markers[(markers > (ith_min_col * b_nr - 1)) & (markers < ith_max_col * b_nr)]

                if len(markers_in_block) > 0:
                    # adding links between markers and ext_node to the graph
                    g = add_marker(g, id_ext_node, markers_in_block)
                else:
                    g = resize_graph(g, (id_ext_node + 1, id_ext_node + 1))

                if front_nodes is not None:
                    # we have to add a node in the frontier
                    front_nodes = np.concatenate([front_nodes, [id_ext_node]])

            if return_map:
                min_val_node = b_nr * ith_min_col
                max_val_node = b_nr * (ith_min_col + b_nc) if i < max_it - 1 else nr*nc
                # map that associate at each pixel in the ith image a node in the graph g
                map_px_to_nodes = np.arange(min_val_node, max_val_node)

                if i < max_it - 1:
                    map_px_to_nodes = map_px_to_nodes.reshape((b_nr, b_nc), order='F')
                else:
                    map_px_to_nodes = map_px_to_nodes.reshape((b_nr, nc - ith_min_col), order='F')
            # updating min col for the next iteration
            ith_min_col = ith_max_col - 1

            if return_map:
                yield (img, map_px_to_nodes), g, root, front_nodes
            else:
                yield img, g, root, front_nodes


class VerticalStreaming(AbstractImageStreamingGenerator):
    def generate_stream(self, block_shape, markers=None, return_map=False):
        """
        Function that generate a vertical stream of image all with the same size.

        Parameters
        ----------
        block_shape: tuple
            Size of the block image

        markers: list
            List of markers in the image

        return_map: bool (default False)
            If true the method should yields also an additional image that represent a map between image pixels and
            nodes of the graph associated to the image.

        """

        # image size
        nr, nc, nz = self.img.shape

        b_nr, b_nc = block_shape

        # we compute the number of iterations
        max_it = np.ceil((nr - b_nr) / (b_nr - 1)).astype(int) + 1

        # initilizing ith_min_row
        ith_min_row = 0

        # additional input parameter
        all_markers = markers

        id_ext_node = nr*nc

        for i in range(max_it):
            # computing ith max row
            ith_max_row = min((b_nr * (i + 1)) - i, nr)

            # selecting ith img in the streaming
            img = self.img[ith_min_row:ith_max_row, :b_nc]

            # graph associated to ith block
            g = img_to_graph(img, row_offset=ith_min_row)

            # id root node for the ith tree
            root = b_nc * (ith_max_row - 1) if i < max_it - 1 else None

            # id nodes front for the ith iteration
            front_nodes = np.arange(b_nc * (ith_max_row - 1), b_nc * ith_max_row, dtype=np.int32) if i < max_it - 1 \
                else None

            if all_markers is not None:  # case segmentation with markers
                # fetching markers in current image block
                markers_in_block = all_markers[(all_markers > (ith_min_row * b_nc - 1)) & (all_markers < ith_max_row* b_nc)]

                if len(markers_in_block) > 0:
                    # adding links between markers and ext_node to the graph
                    g = add_marker(g, id_ext_node, markers_in_block)
                else:
                    g = resize_graph(g, (id_ext_node + 1, id_ext_node + 1))

                if front_nodes is not None:
                    # we have to add a node in the frontier
                    front_nodes = np.concatenate([front_nodes, [id_ext_node]])

            if return_map:
                min_val_node = b_nc * ith_min_row
                max_val_node = b_nc * (ith_min_row + b_nr) if i < max_it - 1 else nr * nc

                # map that associate at each pixel in the ith image a node in the graph g
                map_px_to_nodes = np.arange(min_val_node, max_val_node)

                if i < max_it:
                    map_px_to_nodes = map_px_to_nodes.reshape((b_nr, b_nc))
                else:
                    map_px_to_nodes = map_px_to_nodes.reshape((nr - ith_min_row, b_nc))

            # updating ith min row
            ith_min_row = ith_max_row - 1

            # yielding img
            if return_map:
                yield (img, map_px_to_nodes), g, root, front_nodes
            else:
                yield img, g, root, front_nodes


def add_marker(g, id_ext_node, index_markers, gshape=None):
    """
    Function that link edges between the external node and the markers

    Parameters
    ----------
    g: NxN csr_matrix
        adjacency matrix of a graph g

    id_ext_node: int
        id for the external node (i.e. id of the well in the Watershed algorithm on graph)

    index_markers: M ndarray
        arrays containing the ids of the markers

    gshape: tuple
        shape of the graph g

    Returns
    -------
    G: csr_matrix
        adjacency matrix of modified graph
    """

    # implemented by Santiago-Velasco-Forero
    if gshape is None:
        # by default we define gshape as the max between
        #  the shape of the graph and the id of external node
        gsize = max(id_ext_node+1, g.shape[0])
        gshape = (gsize, gsize)

    [ix, iy, v] = find(g)
    # selecting markers to add
    # ix_markers = np.intersect1d(ix, index_markers)
    # iy_markers = np.intersect1d(iy, index_markers)
    # markers_to_add = np.union1d(ix_markers, iy_markers)

    # adding edges that connect the markers with the external node
    ix = np.concatenate([ix, index_markers])

    iy = np.concatenate([iy, np.ones(len(index_markers)) * id_ext_node])

    v = np.concatenate([v, -np.ones(len(index_markers))])

    # G is the new graph
    G = csr_matrix((v, (ix, iy)), shape=gshape)

    return G
