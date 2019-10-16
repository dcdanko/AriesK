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
    if isinstance(path, str):
        path = [path]
    lang = 'c++'
    return Extension(
        name,
        path,
        include_dirs=[numpy.get_include()],
        extra_compile_args=extra_compile_args,
        language=lang,
    )


extensions = [
    make_ext(el) for el in [
        ('ariesk/utils/bloom_filter.pyx', 'ariesk.utils.bloom_filter'),
        ('ariesk/utils/kmers.pyx', 'ariesk.utils.kmers'),
        ('ariesk/utils/dists.pyx', 'ariesk.utils.dists'),

        ('ariesk/grid_builder.pyx', 'ariesk.grid_builder'),
        ('ariesk/grid_searcher.pyx', 'ariesk.grid_searcher'),
        ('ariesk/contig_searcher.pyx', 'ariesk.contig_searcher'),
        ('ariesk/linear_searcher.pyx', 'ariesk.linear_searcher'),
        ('ariesk/dbs/kmer_db.pyx', 'ariesk.dbs.kmer_db'),
        ('ariesk/dbs/core_db.pyx', 'ariesk.dbs.core_db'),
        ('ariesk/dbs/contig_db.pyx', 'ariesk.dbs.contig_db'),
        ('ariesk/dbs/pre_contig_db.pyx', 'ariesk.dbs.pre_contig_db'),
        ('ariesk/pre_db.pyx', 'ariesk.pre_db'),
        ('ariesk/cluster.pyx', 'ariesk.cluster'),
        ('ariesk/ram.pyx', 'ariesk.ram'),

        ('ariesk/seed_align.pyx', 'ariesk.seed_align'),
        #('ariesk/ssw.pyx', 'ariesk.ssw'),
    ]
] + [
    Extension(
        'ariesk.ssw',
        ['ariesk/ssw.pyx'],
        include_dirs=[numpy.get_include(), 'ariesk/_lib/ssw.c'],
        extra_compile_args=extra_compile_args,
        extra_link_args=['-Lariesk/_lib/ssw.c'],
        language='c++',
    ),
    # Extension(
    #     'ariesk.lib_ssw',
    #     ['ariesk/_lib/ssw.c'],
    #     include_dirs=[numpy.get_include(), 'ariesk/_lib/ssw.c'],
    #     extra_compile_args=["-O3", "-ffast-math", "-march=native", "-fopenmp", '-Wno-error=declaration-after-statement'],
    #     extra_link_args=['-Lariesk/_lib/ssw.c'],
    #     language='c',
    # )

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
    ext_modules=cythonize(extensions),
)
