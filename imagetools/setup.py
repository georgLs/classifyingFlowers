# -*- coding: utf-8 -*-
"""
Created on Sat May 25 23:45:40 2019

@author: Aaron
"""

from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(ext_modules = cythonize('analysis.pyx'),
      include_dirs=[numpy.get_include()])
