from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABC, abstractmethod
import numpy as np

from SST.utils import img_to_graph


class AbstractImageStreamingGenerator(ABC):
    """
    Abstract class that we will use create several generator starting from images
    """
    def __init__(self, img):
        self.img = np.atleast_3d(img)
        super(AbstractImageStreamingGenerator, self).__init__()

    @abstractmethod
    def generate_stream(self, block_shape, **kwargs):
        pass


class HorizontalStreaming(AbstractImageStreamingGenerator):
    def generate_stream(self, block_shape, **kwargs):
        """
        Function that generate an Horizontal stream of image all with the same size.

        Parameters
        ----------
        block_shape: tuple
            Size of the image block

        kwargs: dict
            additional parameters

        """
        # image size
        nr, nc, nz = self.img.shape

        # size of streaming block
        b_nr, b_nc = block_shape

        # we compute the number of iterations
        max_it = np.ceil((nc - b_nc) / (b_nc - 1)).astype(np.int) + 1

        # initializing min_col for the first iteration
        min_col = 0

        for i in range(max_it):
            # max col of the ith img
            max_col = min((b_nc * (i+1)) - i, nc)

            # getting ith block
            img = self.img[:b_nr, min_col:max_col]

            # case for the grayscale image
            if nz == 1:
                img = img[:, :, 0]

            # getting graph associated to ith image block
            g = img_to_graph(img, order='F', col_offset=min_col)

            # id root node for the ith tree
            root = b_nr * (max_col - 1) if i < max_it - 1 else None

            # id nodes front for the ith iteration
            front_nodes = np.arange(b_nr * (max_col - 1), b_nr * max_col, dtype=int) if i < max_it - 1 else None

            # updating min col for the next iteration
            min_col = max_col - 1

            yield img, g, root, front_nodes


class VerticalStreaming(AbstractImageStreamingGenerator):
    def generate_stream(self, block_shape, **kwargs):
        """
        Function that generate an Horizontal stream of image all with the same size.

        Parameters
        ----------
        block_shape: tuple
            Size of the image block

        kwargs: dict
            additional parameters

        """

        # image size
        nr, nc, nz = self.img.shape

        b_nr, b_nc = block_shape

        # we compute the number of iterations
        max_it = np.ceil((nr - b_nr) / (b_nr - 1)).astype(int) + 1

        # initilizing ith_min_row
        ith_min_row = 0

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

            # updating ith min row
            ith_min_row = ith_max_row - 1
            # yielding img
            yield img, g, root, front_nodes
