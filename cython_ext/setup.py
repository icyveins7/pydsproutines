# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 15:22:10 2021

@author: Seo
"""

from setuptools import setup, Extension

from Cython.Build import cythonize

import numpy

extensions = [
    Extension("PyViterbiDemodulator", ["PyViterbiDemodulator.pyx"],
              include_dirs=[numpy.get_include()],
              libraries=["ippcore", "ipps"],
              library_dirs=[],
              language = "c++")
    ]

setup(ext_modules=cythonize(extensions,
                            compiler_directives={'language_level' : 3}))