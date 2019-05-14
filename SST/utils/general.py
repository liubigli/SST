import numpy as np

# Generalized N-dimensional products
def cartesian_product(arrays):
    """
    Parameters
    ----------
    arrays: list of arrays

    Returns
    -------
    cp: ndarray
        Cartesian product of all arrays in list
    """
    # lenght of list of arrays
    la = len(arrays)
    a = np.atleast_3d(arrays)

    dtype = np.find_common_type([a.dtype for a in arrays], [])

    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)

    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a

    cp = arr.reshape(-1, la)

    return cp
