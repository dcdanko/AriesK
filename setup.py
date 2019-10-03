#!/usr/bin/env python
# -*- coding: utf-8 -*-

import setuptools
from Cython.Build import cythonize
import numpy
from distutils.extension import Extension

extra_compile_args = ['-std=c++11', "-O3", "-ffast-math", "-march=native", "-fopenmp" ]
extra_link_args = ['-fopenmp']


def make_ext(args):
    path, name = args
    return Extension(
        name,
        [path],
        include_dirs=[numpy.get_include()],
        extra_compile_args=extra_compile_args,
        language='c++',
    )


extensions = [
    make_ext(el) for el in [
        ('ariesk/utils/bloom_filter.pyx', 'ariesk.utils.bloom_filter'),
        ('ariesk/utils/kmers.pyx', 'ariesk.utils.kmers'),
        ('ariesk/utils/dists.pyx', 'ariesk.utils.dists'),

        ('ariesk/grid_builder.pyx', 'ariesk.grid_builder'),
        ('ariesk/grid_searcher.pyx', 'ariesk.grid_searcher'),
        ('ariesk/linear_searcher.pyx', 'ariesk.linear_searcher'),
        ('ariesk/db.pyx', 'ariesk.db'),
        ('ariesk/cluster.pyx', 'ariesk.cluster'),
        ('ariesk/ram.pyx', 'ariesk.ram'),
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
    package_dir={
        'ariesk': 'ariesk',
    },
    install_requires=[
        'click',
        'pandas',
        'scipy',
        'numpy',
        'umap-learn',
    ],
    entry_points={
        'console_scripts': [
            'ariesk=ariesk.cli:main',
            'inquest-ariesk=ariesk.inquest.cli:main',
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
