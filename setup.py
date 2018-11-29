from distutils.core import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy as np

ext_modules = [
    Extension("SST.streaming.graph._traverse",
              ["SST/streaming/graph/_traverse.pyx"])
]

setup(
    name="SST",
    cmdclass = {'build_ext': build_ext},
    ext_modules = cythonize(ext_modules),
    include_dirs = [np.get_include()]
)
