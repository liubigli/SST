from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import time
import numpy as np
from numpy import matlib
from scipy.sparse import csr_matrix, find
from matplotlib import pyplot as plt


def pixel_to_node(p, ishape, order='C'):
    """
    From pixel coordinates to associated node in the image graph

    Parameters
    ----------
    p: tuple
        Coordinates (i,j) of the pixel

    ishape: tuple
        Size of the image

    order : {'C', 'F', 'A'}, optional
        Read the elements of `a` using this index order, and place the
        elements into the reshaped array using this index order.  'C'
        means to read / write the elements using C-like index order,
        with the last axis index changing fastest, back to the first
        axis index changing slowest. 'F' means to read / write the
        elements using Fortran-like index order, with the first index
        changing fastest, and the last index changing slowest. Note that
        the 'C' and 'F' options take no account of the memory layout of
        the underlying array, and only refer to the order of indexing.
        'A' means to read / write the elements in Fortran-like index
        order if `a` is Fortran *contiguous* in memory, C-like order
        otherwise.

    Returns
    -------
    node: int
        Id of the corresponding node in the graph associated to image

    """

    if order == 'C':  # C-like index order
        return p[0] * ishape[1] + p[1]

    if order == 'F':  # Fortran-like index order
        return p[0] + p[1] * ishape[0]


def node_to_pixel(n, ishape, order='C'):
    """
    From node in image graph to associated pixel

    Parameters
    ----------
    n: int
        Id of the corresponding node in the graph associated to image

    ishape: tuple
        Size of the image

    order : {'C', 'F', 'A'}, optional
        Read the elements of `a` using this index order, and place the
        elements into the reshaped array using this index order.  'C'
        means to read / write the elements using C-like index order,
        with the last axis index changing fastest, back to the first
        axis index changing slowest. 'F' means to read / write the
        elements using Fortran-like index order, with the first index
        changing fastest, and the last index changing slowest. Note that
        the 'C' and 'F' options take no account of the memory layout of
        the underlying array, and only refer to the order of indexing.
        'A' means to read / write the elements in Fortran-like index
        order if `a` is Fortran *contiguous* in memory, C-like order
        otherwise.

    Returns
    -------
    node: tuple
        Coordinates of the corresponding pixel in the image

    """

    i, j = 0, 0  # initializing returning variables

    if order == 'C':  # C-like index order
        i = np.floor(n / ishape[1]).astype(int)
        j = n % ishape[1]

    if order == 'F':  # Fortran-like index order
        i = n % ishape[0]
        j = np.floor(n / ishape[0]).astype(int)

    return i, j


def _img_to_4_connected_graph(img):
    """
    Function that returns the 4 connected weighted graph associated to the input image.
    The weights of the edges are the differences between the pixels.
    This is a simpler version of img_to_graph function implemented below

    Parameters
    ----------
    img: ndarray
        input image

    Returns
    -------
    graph: csr_matrix
        graph associated to image

    """
    # implemented by Santiago-Velasco-Forero

    from sklearn.feature_extraction.image import _make_edges_3d
    # convert input image to 3d image
    img = np.atleast_3d(img)
    # get image shape
    nr, nc, nz = img.shape
    # defining little eps in order to have also zero values edges
    eps = 1e-10
    # get edges for
    edges = _make_edges_3d(nr, nc, n_z=1)

    if nz == 1:
        imgtemp = img.flatten().astype('float')
        # consider 4 connectivity
        grad = abs(imgtemp[edges[0]] - imgtemp[edges[1]]) + eps
    else:
        # in case of coloured images we use the maximum of color differences
        # among all channels as the edge weights
        # we copy images
        imgtemp = img.reshape(-1, nz).astype('float')
        grad = np.abs(imgtemp[edges[0]] - imgtemp[edges[1]]) + 1e-10
        grad = grad.max(axis=1)

    return csr_matrix((grad, (edges[0], edges[1])), shape=(nr * nc, nr * nc))


def img_to_graph(img, metric=None, order='C', **kwargs):
    """
    Function that return a 4-connected graph from an image on which the weights are defined according to a given metric

    Parameters
    ----------
    img: (N,M) ndarray
        input image

    metric: function (default None)
        metric used to assign weights to edges

    order : {'C', 'F', 'A', 'K'}, optional
        'C' means to flatten in row-major (C-style) order.
        'F' means to flatten in column-major (Fortran-style) order.
        'A' means to flatten in column-major order if `a` is Fortran *contiguous* in memory, row-major order otherwise.
        'K' means to flatten `a` in the order the elements occur in memory.
        The default is 'C'.

    kwargs: dict
        supplemental arguments for the metric function

    Returns
    -------
    graph: (N*M, N*M) csr_matrix
        adjacency matrix of the graph associated to image img
    """
    # implemented by Santiago-Velasco-Forero
    from sklearn.feature_extraction.image import _make_edges_3d

    img = np.atleast_3d(img)

    nr, nc, nz = img.shape

    # default case: we build a 4-connected graph
    if order == 'C':  # order for image pixels K=columns-major order/ C=row-major order
        edges = _make_edges_3d(nr, nc, n_z=1)
    else:
        edges = _make_edges_3d(nc, nr, n_z=1)

    # we copy images
    imgtemp = img.copy().astype('float')
    # we flatten only first two dimension and imgtemp is a {nr*nc} X {nz} vector
    if nz == 1:
        imgtemp = imgtemp.flatten(order=order)
    else:
        imgtemp = imgtemp.reshape(-1, nz, order=order)

    if metric is None:
        # default metric is the gradient metric
        weights = np.abs(imgtemp[edges[0]] - imgtemp[edges[1]]) + 1e-10
        if nz > 1:
            # in case of color/multispectral images we take as distance the max distance in all channels
            weights = weights.max(axis=1)
        else:
            # gray level image
            weights = weights.flatten()
    else:
        # otherwise is possible to pass a custom function to compute distances between two adjacent pixels
        weights = np.array([metric(imgtemp, e[0], e[1], **kwargs) for e in edges.T])

    # to save space we delete temp image
    del imgtemp

    # is possible to define a custom mask and remap edges according to this map
    mask = kwargs.get('mask', None)

    if mask is not None:
        # graph shape is determined by the max value of mask
        g_shape = (mask.max() + 1, mask.max() + 1)

        # in case of custom mask we return the graph
        return csr_matrix((weights, (mask[edges[0]], mask[edges[1]])), shape=g_shape)

    # otherwise is possible to define an offset on the graph in order to manage the case of split images
    # the offset is defined according to the order of flattening of the image.
    # So is always defined in a unique direction and it allows us to define a graph of the good dimension
    g_col = nr * nc

    offset = 0
    # switch in case of row-major or column-major order
    if order == 'C':
        # case in which the matrix is flatten in c-style i.e. row-major order
        row_offset = kwargs.get('row_offset', 0)
        offset = row_offset * nc
        # updating graph dimension
        g_col = (nr + row_offset) * nc

    elif order == 'F':
        # case in which the matrix is flatten in fortran-style i.e. columns-major order
        col_offset = kwargs.get('col_offset', 0)
        offset = col_offset * nr
        g_col = nr * (nc + col_offset)

    # graph associated to images are always represented as square sparse matrices
    g_shape = (g_col, g_col)

    # we return the computed graph as csr_matrix
    return csr_matrix((weights, (offset + edges[0], offset + edges[1])), shape=g_shape)


def plot_graph(img,
               graphs,
               labels=None,
               filename="",
               figsize=(8, 8),
               order='C',
               saveplot=False,
               colors=None):
    """
    Function that plots up to 5 graphs contained in the list graphs

    Parameters
    ----------
    img: ndarray
        Image to plot

    graphs: list
        Graphs to plot

    labels: list
        list of labels that remaps nodes in list of graphs

    filename: string
        Name of the file to save

    figsize: tuple
        Dimension of the figure to save

    order : {'C', 'F', 'A', 'K'}, optional
        'C' means to flatten in row-major (C-style) order.
        'F' means to flatten in column-major (Fortran-style) order.
        'A' means to flatten in column-major order if `a` is Fortran *contiguous* in memory, row-major order otherwise.
        'K' means to flatten `a` in the order the elements occur in memory.
        The default is 'C'

    saveplot: bool
        Set to True to save the file. Default is False.

    colors: list
        list of colors for graphs. Default ['g', 'r', 'b', 'k', 'm'].
    """
    # thanks to an idea of Santiago-Velasco-Forero
    if type(graphs) is not list:
        graphs = [graphs]

    if labels is not None and type(labels) is not list:
        labels = [labels]

    if colors is None:
        colors = ['g', 'r', 'b', 'k', 'm']

    nr, nc = img.shape[:2]
    n_plots = min(len(graphs), len(colors))
    edges_list = [[]] * n_plots
    for i in range(n_plots):
        edges_list[i] = find(graphs[i])
    if order == 'F':
        dx = np.matlib.repmat(np.arange(nc), nr, 1).transpose()
        dy = np.matlib.repmat(np.arange(nr), nc, 1)
    else:
        dx = np.matlib.repmat(np.arange(nr), nc, 1).transpose()
        dy = np.matlib.repmat(np.arange(nc), nr, 1)
    dx = dx.flatten()
    dy = dy.flatten()
    plt.figure(figsize=figsize)
    plt.gcf()
    plt.gca()

    plt.imshow(img)
    plt.tight_layout(pad=0)
    plt.axis('off')

    for i in range(n_plots):
        if labels is not None:
            imi, jmi = labels[i][edges_list[i][0]], labels[i][edges_list[i][1]]
        else:
            imi, jmi = edges_list[i][0], edges_list[i][1]
        if order == 'F':
            plt.plot([dx[imi], dx[jmi]], [dy[imi], dy[jmi]], '-' + colors[i])
        else:
            plt.plot([dy[imi], dy[jmi]], [dx[imi], dx[jmi]], '-' + colors[i])

    if saveplot:
        plt.savefig(filename + '_' + time.strftime('%s') + ".png", bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close()
    else:
        plt.show()

    plt.show()


def plot_sub_graph(img, graph, min_row, max_row, min_col, max_col, figsize=(8, 8), order='C', colors=None):
    """
    Function that plot a subgraph of the main graph

    Parameters
    ----------
    img: ndarray
        input image

    graph: csr_matrix
        input graph

    min_col: int
        minimum value of x interval

    max_col: int
        max value for the x interval

    min_row: int
        min value for the y interval

    max_row: int
        max value for the y interval

    figsize: tuple
        Size of the figure

    colors: list
        colors to use for subgraph

    order : {'C', 'F', 'A', 'K'}, optional
        'C' means to flatten in row-major (C-style) order.
        'F' means to flatten in column-major (Fortran-style) order.
        'A' means to flatten in column-major order if `a` is Fortran *contiguous* in memory, row-major order otherwise.
        'K' means to flatten `a` in the order the elements occur in memory.
        The default is 'C'.
    """
    # the basic idea of this function is to select pixels contained in the interval [ymin:ymax, xmin:xmax]
    # and the corresponding nodes and edges in the graphs. Once selected we call the function plot_graph
    nr, nc = img.shape[:2]  # getting image shapes

    # selecting subimg
    subimg = img[min_row:max_row, min_col:max_col]

    # getting subimage dimensions
    # snr, snc = subimg.shape[:2]

    # selecting nodes corresponding to the pixels in the graph
    graph_nodes = np.arange(nr * nc).reshape(nr, nc, order=order)[min_row:max_row, min_col:max_col].flatten(order=order)
    # # cartesian product of the nodes ids
    # cart = np.array(np.meshgrid(graph_nodes, graph_nodes)).T.reshape(-1, 2)

    # selecting subgraph
    # sub = csr_matrix(graph[cart[:, 0], cart[:, 1]].reshape(snr * snc, snr * snc))
    if type(graph) is list:
        sub = []
        for g in graph:
            g_rows, g_cols = g.shape
            sub.append(g[graph_nodes[graph_nodes < g_rows], :][:, graph_nodes[graph_nodes < g_cols]])
    else:
        g_rows, g_cols = graph.shape
        sub = graph[graph_nodes[graph_nodes < g_rows], :][:, graph_nodes[graph_nodes < g_cols]]
    # plotting subimage and subgraph
    plot_graph(subimg, sub, figsize=figsize, order=order, colors=colors)


def accumarray(indices, vals, size, func='plus', fill_value=0):
    """
    from: https://github.com/pathak22/videoseg/blob/master/src/utils.py
    Implementing python equivalent of matlab accumarray.
    Taken from SDS repo: master/superpixel_representation.py#L36-L46

    Parameters
    ----------
    indices: ndarray
        must be a numpy array (any shape)

    vals: ndarray
        numpy array of same shape as indices or a scalar

    size: int
        must be the number of diffent values

    func: {'plus', 'minus', 'times', 'max', 'min', 'and', 'or'} optional
        Default is 'plus'

    fill_value: int
        Default is 0

    """

    # get dictionary
    function_name_dict = {
        'plus': (np.add, 0.),
        'minus': (np.subtract, 0.),
        'times': (np.multiply, 1.),
        'max': (np.maximum, -np.inf),
        'min': (np.minimum, np.inf),
        'and': (np.logical_and, True),
        'or': (np.logical_or, False)}

    if func not in function_name_dict:
        raise KeyError('Function name not defined for accumarray')

    if np.isscalar(vals):
        if isinstance(indices, tuple):
            shape = indices[0].shape
        else:
            shape = indices.shape
        vals = np.tile(vals, shape)

    # get the function and the default value
    (func, value) = function_name_dict[func]

    # create an array to hold things
    output = np.ndarray(size)
    output[:] = value
    func.at(output, indices, vals)

    # also check whether indices have been used or not
    isthere = np.ndarray(size, 'bool')
    istherevals = np.ones(vals.shape, 'bool')
    (func, value) = function_name_dict['or']
    isthere[:] = value
    func.at(isthere, indices, istherevals)

    # fill things that were not used with fill value
    output[np.invert(isthere)] = fill_value

    return output


def label_image(img, labels, order='C'):
    """
    Function that given an image and a vector of labels for its pixels returns the corresponding segmented image

    Parameters
    ----------
    img: nxm ndarray
        input image
    labels: n*m ndarray
        Array of labels for pixels of the input image.

    order : {'C', 'F', 'A', 'K'}, optional
        'C' means to flatten in row-major (C-style) order.
        'F' means to flatten in column-major (Fortran-style) order.
        'A' means to flatten in column-major order if `a` is Fortran *contiguous* in memory, row-major order otherwise.
        'K' means to flatten `a` in the order the elements occur in memory.
        The default is 'C'.

    Returns
    -------
    img_label: nxm ndarray
        Resulting segmented image
    """
    img = np.atleast_3d(img)
    # image dimensions
    nr, nc, nz = img.shape

    n_cc = labels.max() + 1

    s = []
    for i in range(nz):
        s.append(accumarray(labels, img[:, :, i].flatten(order=order), n_cc, func='plus'))

    ne = accumarray(labels, np.ones(nr*nc), n_cc, func='plus')

    for i in range(nz):
        s[i] = s[i] / ne
        s[i] = (s[i][labels]).reshape((nr, nc), order=order)

    img_label = np.zeros(img.shape)

    for i in range(nz):
        img_label[:, :, i] = s[i]

    if nz == 1:
        return img_label[:, :, 0]
    else:
        return img_label


def stick_two_images(img1, img2, num_overlapping=0, direction='H'):
    """
    Function that sticks two different images

    img1: ndarray
        First image to stick

    img2: ndarray
        Second image to stick

    num_overlapping: int
        number of overlapping rows or columns

    direction: {'H', 'V'} optional
        Stick direction.
        'H' means horizontal direction of sticking, i.e. the images are one near the other
        'V' means vertical direction of sticking, i.e. the images are one above the other

    Returns
    -------
    merged_img: MxN ndarray

    """
    img1 = np.atleast_3d(img1)
    img2 = np.atleast_3d(img2)

    # getting shape of the two images
    nr1, nc1, nz1 = img1.shape
    nr2, nc2, nz2 = img2.shape

    if direction.lower() == 'h':
        if nr1 != nr2 or nz1 != nz2:
            raise ValueError('Dimension mismatch! The two images have a different number of rows or channels')

        merged_img = np.zeros((nr1, nc1 + nc2 - num_overlapping, nz1), dtype=img1.dtype)
        merged_img[:, :nc1] = img1
        merged_img[:, nc1 - num_overlapping:] = img2

        if nz1 > 1:
            return merged_img
        else:
            return merged_img[:, :, 0]

    if direction.lower() == 'v':
        if nc1 != nc2 or nz1 != nz2:
            raise ValueError('Dimension mismatch! The two images have a different number of rows or channels')

        merged_img = np.zeros((nr1 + nr2 - num_overlapping, nc1, nz1), dtype=img1.dtype)
        merged_img[:nr1, :] = img1
        merged_img[nr1 - num_overlapping:, :] = img2

        if nz1 > 1:
            return merged_img
        else:
            return merged_img[:, :, 0]

    else:
        raise ValueError('Direction of merging not known')


def add_circle(img, center, radius):
    from PIL import Image, ImageDraw
    new_img = Image.fromarray(img)
    draw = ImageDraw.Draw(new_img)
    draw.ellipse((center[0]-radius, center[1]-radius, center[0]+radius, center[1] + radius),
                 fill='white', outline='white')
    img = np.array(new_img)
    return img


def generate_random_pois(shape, n_circles, return_size_radii=False):
    """

    Parameters
    ----------
    shape: tuple
        image shape
    n_circles: int
        number of circles to
    return_size_radii:

    Return
    ------
    """
    nr, nc = shape[:2]
    # in order to avoid empty circles we put a frame of 10% of image size where the center of any
    # circle cannot fall into
    r_frame = nr // 10
    c_frame = nc // 10
    r_centers = np.random.randint(r_frame, nr - r_frame, size=n_circles)
    c_centers = np.random.randint(c_frame, nc - c_frame, size=n_circles)
    centers = np.c_[r_centers, c_centers]
    radii = np.zeros(n_circles)

    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(centers)
    distances, indices = nbrs.kneighbors(centers)
    img = np.zeros((nr, nc), dtype=np.uint8)
    for i in range(n_circles):
        max_radius = min(centers[i, 0], centers[i, 1], nr - centers[i, 0], nc - centers[i, 1], distances[i, 1]) // 2
        if max_radius > 1:
            radii[i] = np.random.randint(1, max_radius)
        else:
            radii[i] = 1

        img = add_circle(img, centers[i], radii[i])

    if return_size_radii:
        return img, radii

    return img


def generate_pathological_bad_example(shape):
    """
    Utils functions that generate a pathological example for our streaming algorithm that will make our
    algorithm explode. Basically a particular case when the unstable edges in the MST
    are at each iteration all the image graph. The final MST can be represented as follow
      |<----------------------------------------------- IMAGE COLUMNS ----------------------------------------------->|

    = x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x
    I |
    M |
    A |
    G x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x
    E |
      |
    R |
    O x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x
    W |
    S |
      |
    = x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x

    |<-------IMAGE BLOCK------->|<-------IMAGE BLOCK------->|<-------IMAGE BLOCK------->|<-------IMAGE BLOCK------->|

    In order to obtain this MST we need an image with special values. Essentially, the image is constant along the rows
    except for the pixels in the first column. We choose pixel's values in the first column in order that weights of
    edges between two pixels in the first columns is smaller than any weight between two pixels in different rows.

              n*d
        I(P0) --- I(P1)
          |         |
        d |         | 1
          |         |
        I(P2) --- I(P3)
            (n-1)*d


    Parameters
    ----------
    shape: tuple
        Image size

    Return
    ------
    img: ndarray
        Pathological image that make streaming_spanning_tree methods explode
    """

    nr, nc = shape[:2]

    img = np.repeat(np.arange(nr), nc).reshape(nr, nc)
    img = img.astype(np.float64)
    # getting n
    n = nr // 2
    # small delta
    delta = 1 / (2 * n)
    it = np.arange(n, 0, -1)

    if nr % 2 == 0:
        img[:n, 0] += it*delta + delta/2
        img[n:, 0] += (-it[::-1]*delta - delta/2)
    else:
        img[:n, 0] += (it * delta)
        img[(n+1):, 0] += (-it[::-1]*delta)

    return img


def generate_pathological_good_example(shape):
    """
    Function that returns a particularly good image for streaming_spanning_tree methods. In fact this image is constant
    along the columns and the edges_in_cycles during the streaming operation is the smallest possible graph.

    Parameters
    ----------
    shape: tuple
        Shape of the output image

    Returns
    -------
    img: ndarray
        Pathological example that works perfectly for streaming_spanning_tree methods
    """

    nr, nc = shape[:2]
    n = nc // 2
    k = nc % 2
    # the image that we generate contains only 0 and 1
    img = np.repeat(np.concatenate((n*[0, 1], k*[0])), nr).reshape((nr, nc), order='F')

    return img
