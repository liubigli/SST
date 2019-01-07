import os
import numpy as np
from scipy.misc import imread
from config import PROJECT_ROOT
from SST.streaming.streaming_generators import HorizontalStreaming
from SST.applications.segmentation.streaming_segmentation import  quasi_flat_zone_streaming

def test_qfz_streaming():
    # loading image
    img = imread(os.path.join(PROJECT_ROOT, 'examples', 'grains.png'))
    # img shape
    nr, nc = img.shape[:2]

    stream_ncc = np.array([3, 5, 7])

    # initializing streaming
    gen = HorizontalStreaming(img)
    stream = gen.generate_stream(block_shape=(nr, 100))
    stable_labels = None
    for n, labels in enumerate(quasi_flat_zone_streaming(stream, 10)):

        if n == 0:
            stable_labels = labels
        else:
            stable_labels = np.concatenate((stable_labels, labels), axis=1)

        # at the first iteration in image grains the number of stable connected components must be 3
        # at the second must be 5
        # at the third and final iteration the number of stable connected components must be 7
        assert np.unique(stable_labels[1]).shape[0] == stream_ncc[n]