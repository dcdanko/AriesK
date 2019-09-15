#!/usr/bin/env python
# -*- coding: utf-8 -*-

import setuptools
from Cython.Build import cythonize
import numpy
from distutils.extension import Extension

extra_compile_args = ['-std=c++11', "-O3", "-ffast-math", "-march=native", "-fopenmp" ]
extra_link_args = ['-fopenmp']

extensions = [
    Extension(
        'ariesk.dists',
        ['ariesk/dists.pyx'],
        include_dirs=[numpy.get_include()],
        extra_compile_args=extra_compile_args,
        language='c++',
    ),
    Extension(
        'ariesk.ram',
        ['ariesk/ram.pyx'],
        include_dirs=[numpy.get_include()],
        extra_compile_args=extra_compile_args,
        language='c++',
    ),
    Extension(
        'ariesk.grid_cover',
        ['ariesk/grid_cover.pyx'],
        include_dirs=[numpy.get_include()],
        extra_compile_args=extra_compile_args,
        language='c++',
    ),
    Extension(
        'ariesk.searcher',
        ['ariesk/searcher.pyx'],
        include_dirs=[numpy.get_include()],
        extra_compile_args=extra_compile_args,
        language='c++',
    ),
    Extension(
        'ariesk.db',
        ['ariesk/db.pyx'],
        include_dirs=[numpy.get_include()],
        extra_compile_args=extra_compile_args,
        language='c++',
    ),
    Extension(
        'ariesk.utils',
        ['ariesk/utils.pyx'],
        include_dirs=[numpy.get_include()],
        extra_compile_args=extra_compile_args,
        language='c++',
    ),
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
