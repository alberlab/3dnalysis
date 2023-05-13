#!/usr/bin/env python
from distutils.core import setup, Extension

import numpy
import sys

# Add include and library directories from conda envs for swig.
std_include = [sys.prefix + '/include', sys.prefix + '/Library/include']
std_library = [sys.prefix + '/lib', sys.prefix + '/Library/lib']

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

setup(
    name='genome3danalysis',
    version='0.0.0+strucfeat',
    author='Francesco Musella, Ye West',
    author_email='fmusella@g.ucla.edu',
    url='https://github.com/alberlab/genome3danalysis',
    description='3D Genome Analysis Tools',
    packages=['genome3danalysis'],
    # tests_require=tests_require,
    # extras_require=extras_require,
    include_dirs=[numpy_include]+std_include
)
