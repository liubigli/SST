from scipy.misc import ascent
from SST.streaming.streaming_generators import HorizontalStreaming, VerticalStreaming
from SST.utils import stick_two_images


def pytest_generate_tests(metafunc):
    # called once per each test function
    funcarglist = metafunc.cls.params[metafunc.function.__name__]
    argnames = sorted(funcarglist[0])
    metafunc.parametrize(argnames, [[funcargs[name] for name in argnames]
            for funcargs in funcarglist])


class TestGenerators(object):
    params = {
        'test_horizontal_stream': [
            dict(
                img=ascent()[250:350],
                block_shape=(100, 100)
            )
        ],
        'test_vertical_stream': [
            dict(
                img=ascent()[:,100:200],
                block_shape=(100, 100)
            )
        ]
    }

    def test_horizontal_stream(self, img, block_shape):
        # getting image shape
        nr, nc = img.shape[:2]
        b_nr, b_nc = block_shape

        # initializing stream
        gen = HorizontalStreaming(img)
        stream = gen.generate_stream(block_shape=block_shape)

        curr_img = None
        for n, (i, g, r, f) in enumerate(stream):
            if n == 0:
                curr_img = i
            else:
                curr_img = stick_two_images(curr_img, i, num_overlapping=1, direction='H')

            # shape of current image
            c_nr, c_nc = curr_img.shape[:2]

            if (c_nc + n) % block_shape[1] == 0:
                assert i.shape[:2] == block_shape
                assert g.shape[0] == c_nr * c_nc
                assert g.nnz == 2 * b_nr * b_nc - b_nr - b_nc
                assert r == c_nr * (c_nc - 1)
            else:  # last iteration
                assert  i.shape[:2] == (block_shape[0], nc - (b_nc * n - n))
                assert g.shape[0] == nr * nc
                assert g.nnz == 2 * i.shape[0] * i.shape[1] - i.shape[0] - i.shape[1]
                assert r == None
                assert f == None


    def test_vertical_stream(self, img, block_shape):
        # getting image shape
        nr, nc = img.shape[:2]
        b_nr, b_nc = block_shape

        # initializing stream
        gen = VerticalStreaming(img)
        stream = gen.generate_stream(block_shape=block_shape)

        curr_img = None
        for n, (i, g, r, f) in enumerate(stream):
            if n == 0:
                curr_img = i
            else:
                curr_img = stick_two_images(curr_img, i, num_overlapping=1, direction='V')

            # shape of current image
            c_nr, c_nc = curr_img.shape[:2]

            if (c_nr + n) % block_shape[0] == 0:
                assert i.shape[:2] == block_shape
                assert g.shape[0] == c_nr * c_nc
                assert g.nnz == 2 * b_nr * b_nc - b_nr - b_nc
                assert r == c_nc * (c_nr - 1)

            else:  # last iteration
                assert  i.shape[:2] == (nr - (b_nr * n - n) , block_shape[1])
                assert g.shape[0] == nr * nc
                assert g.nnz == 2 * i.shape[0] * i.shape[1] - i.shape[0] - i.shape[1]
                assert r == None
                assert f == None