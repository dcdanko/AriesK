#!/usr/bin/env python
# -*- coding: utf-8 -*-

import setuptools
from Cython.Build import cythonize
import numpy
from distutils.extension import Extension

extra_compile_args = ['-std=c++11', "-O3", "-ffast-math", "-march=native", "-fopenmp" ]
extra_link_args = ['-fopenmp']


def make_ext(path):
    name = path.split('.pyx')[0].replace('/', '.')
    return Extension(
        name,
        [path],
        include_dirs=[numpy.get_include()],
        extra_compile_args=extra_compile_args,
        language='c++',
    )


extensions = [
    make_ext(el) for el in [
        'ariesk/ram.pyx',
        'ariesk/grid_cover.pyx',
        'ariesk/searcher.pyx',
        'ariesk/linear_searcher.pyx',
        'ariesk/db.pyx',
        'ariesk/bloom_filter.pyx',
        'ariesk/cluster.pyx',
        'ariesk/utils.pyx',
    ]
]


setuptools.setup(
    name='ariesk',
    version='0.1.0',
    description="Fuzzy K-Mer Phylogeny based on the Ramanujan Fourier Transform",
    author="David C. Danko",
    author_email='dcdanko@gmail.com',
    url='https://github.com/dcdanko/ariesk',
    packages=setuptools.find_packages(),
    package_dir={'ariesk': 'ariesk'},
    install_requires=[
        'click',
        'pandas',
        'scipy',
        'numpy',
        'umap-learn',
    ],
    entry_points={
        'console_scripts': [
            'ariesk=ariesk.cli:main'
        ]
    },
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
    ],
    ext_modules=cythonize(extensions)
)
