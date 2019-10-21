# Copyright Anne M. Archibald 2008
# Additional contributions by Patrick Varilly and Sturla Molden 2012
# Revision by Sturla Molden 2015
# Balanced kd-tree construction written by Jake Vanderplas for scikit-learn
# Released under the scipy license

# distutils: language = c++

from __future__ import absolute_import

import numpy as np
import scipy.sparse

cimport numpy as np
from numpy.math cimport INFINITY

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from libc.string cimport memset, memcpy
from libcpp.vector cimport vector
from libcpp.algorithm cimport sort

cimport cython

from multiprocessing import cpu_count
import threading

cdef extern from "<limits.h>":
    long LONG_MAX

cdef int number_of_processors = cpu_count()

cdef extern from *:
    int NPY_LIKELY(int)
    int NPY_UNLIKELY(int)


# C++ implementations
# ===================

cdef extern from "_lib/kdtree/ckdtree.cpp":
    int ckdtree_isinf(np.float64_t x) nogil

    struct ckdtreenode:
        np.intp_t split_dim
        np.intp_t children
        np.float64_t split
        np.intp_t start_idx
        np.intp_t end_idx
        ckdtreenode *less
        ckdtreenode *greater
        np.intp_t _less
        np.intp_t _greater

    struct ckdtree:
        vector[ckdtreenode]  *tree_buffer
        ckdtreenode   *ctree
        np.float64_t   *raw_data
        np.intp_t      n
        np.intp_t      m
        np.intp_t      leafsize
        np.float64_t   *raw_maxes
        np.float64_t   *raw_mins
        np.intp_t      *raw_indices
        np.float64_t   *raw_boxsize_data
        np.intp_t size

    # External build and query methods in C++.
    
    int build_ckdtree(ckdtree *self,
                         np.intp_t start_idx,
                         np.intp_t end_idx,
                         np.float64_t *maxes,
                         np.float64_t *mins,
                         int _median,
                         int _compact) nogil except +

    int build_weights(ckdtree *self,
                         np.float64_t *node_weights,
                         np.float64_t *weights) nogil except +

    int query_ball_tree(const ckdtree *self,
                           const ckdtree *other,
                           const np.float64_t r,
                           const np.float64_t p,
                           const np.float64_t eps,
                           vector[np.intp_t] **results) nogil except +


cdef class cKDTree:

    property n:
        def __get__(self): return self.cself.n
    property m:
        def __get__(self): return self.cself.m
    property leafsize:
        def __get__(self): return self.cself.leafsize
    property size:
        def __get__(self): return self.cself.size

    def __cinit__(cKDTree self):
        self.cself = <ckdtree * > PyMem_Malloc(sizeof(ckdtree))
        self.cself.tree_buffer = NULL

    def __init__(
        cKDTree self,
        double[:, :] data,
        np.intp_t leafsize=16,
        compact_nodes=False,
        copy_data=False,
        balanced_tree=False,
        logger=None
        ):
        self.logging = False
        if logger is not None:
            self.logging = True
            self.logger = logger

        cdef: 
            np.float64_t [::1] tmpmaxes, tmpmins
            np.float64_t *ptmpmaxes
            np.float64_t *ptmpmins
            ckdtree *cself = self.cself
            int compact, median

        self.data = data
        cself.n = data.shape[0]
        cself.m = data.shape[1]
        cself.leafsize = leafsize

        self.boxsize = None
        self.boxsize_data = None

        self.maxes = np.ascontiguousarray(
            np.amax(self.data, axis=0),
            dtype=np.float64
        )
        self.mins = np.ascontiguousarray(
            np.amin(self.data, axis=0),
            dtype=np.float64
        )
        self.indices = np.ascontiguousarray(np.arange(self.n, dtype=np.intp))

        self._pre_init()

        compact = 1 if compact_nodes else 0
        median = 1 if balanced_tree else 0

        cself.tree_buffer = new vector[ckdtreenode]()

        tmpmaxes = np.copy(self.maxes)
        tmpmins = np.copy(self.mins)

        ptmpmaxes = &tmpmaxes[0]
        ptmpmins = &tmpmins[0]
        if self.logging:
            self.logger('[KD-Tree] Calling C Build Routine...')
        with nogil: 
            build_ckdtree(cself, 0, cself.n, ptmpmaxes, ptmpmins, median, compact)
        if self.logging:
            self.logger('[KD-Tree] Finished C Build Routine.')
        # set up the tree structure pointers
        self._post_init()

    cdef _pre_init(cKDTree self):
        cself = self.cself

        # finalize the pointers from array attributes

        cself.raw_data = <np.float64_t*> &self.data[0, 0]
        cself.raw_maxes = <np.float64_t*> np.PyArray_DATA(self.maxes)
        cself.raw_mins = <np.float64_t*> np.PyArray_DATA(self.mins)
        cself.raw_indices = <np.intp_t*> np.PyArray_DATA(self.indices)

        if self.boxsize_data is not None:
            cself.raw_boxsize_data = <np.float64_t*>np.PyArray_DATA(self.boxsize_data)
        else:
            cself.raw_boxsize_data = NULL

    cdef _post_init(cKDTree self):
        cself = self.cself
        # finalize the tree points, this calls _post_init_traverse

        cself.ctree = cself.tree_buffer.data()

        # set the size attribute after tree_buffer is built
        cself.size = cself.tree_buffer.size()

        self._post_init_traverse(cself.ctree)

    cdef _post_init_traverse(cKDTree self, ckdtreenode *node):
        cself = self.cself
        # recurse the tree and re-initialize
        # "less" and "greater" fields
        if node.split_dim == -1:
            # leafnode
            node.less = NULL
            node.greater = NULL
        else:
            node.less = cself.ctree + node._less
            node.greater = cself.ctree + node._greater
            self._post_init_traverse(node.less)
            self._post_init_traverse(node.greater)

    def __dealloc__(cKDTree self):
        cself = self.cself
        if cself.tree_buffer != NULL:
            del cself.tree_buffer
        PyMem_Free(cself)

    cdef query_ball_tree(cKDTree self, cKDTree other,
                        np.float64_t r, np.float64_t p=2., np.float64_t eps=0):

        cdef:
            vector[np.intp_t] **vvres
            np.intp_t i, j, n, m
            np.intp_t *cur
            list results
            list tmp

        n = self.n

        try:

            # allocate an array of std::vector<npy_intp>
            vvres = (<vector[np.intp_t] **>
                PyMem_Malloc(n * sizeof(void*)))
            if vvres == NULL:
                raise MemoryError()

            memset(<void*> vvres, 0, n * sizeof(void*))

            for i in range(n):
                vvres[i] = new vector[np.intp_t]()

            # query in C++
            with nogil:
                query_ball_tree(self.cself, other.cself, r, p, eps, vvres)

            # store the results in a list of lists
            results = n * [None]
            for i in range(n):
                m = <np.intp_t> (vvres[i].size())
                if NPY_LIKELY(m > 0):
                    tmp = m * [None]
                    with nogil:
                        sort(vvres[i].begin(), vvres[i].end())
                    cur = vvres[i].data()
                    for j in range(m):
                        tmp[j] = cur[0]
                        cur += 1
                    results[i] = tmp
                else:
                    results[i] = []

        finally:
            if vvres != NULL:
                for i in range(n):
                    if vvres[i] != NULL:
                        del vvres[i]
                PyMem_Free(vvres)

        return results

    def _build_weights(cKDTree self, object weights):
        """
        _build_weights(weights)
        Compute weights of nodes from weights of data points. This will sum
        up the total weight per node. This function is used internally.
        Parameters
        ----------
        weights : array_like
            weights of data points; must be the same length as the data points.
            currently only scalar weights are supported. Therefore the weights
            array must be 1 dimensional.
        Returns
        -------
        node_weights : array_like
            total weight for each KD-Tree node.
        """
        cdef: 
            np.intp_t num_of_nodes
            np.float64_t [::1] node_weights
            np.float64_t [::1] proper_weights
            np.float64_t *pnw
            np.float64_t *ppw

        num_of_nodes = self.cself.tree_buffer.size();
        node_weights = np.empty(num_of_nodes, dtype=np.float64)

        # FIXME: use templates to avoid the type conversion
        proper_weights = np.ascontiguousarray(weights, dtype=np.float64)

        pnw = &node_weights[0]
        ppw = &proper_weights[0]

        with nogil:
            build_weights(self.cself, pnw, ppw)

        return node_weights

    # ----------------------
    # pickle
    # ----------------------

    def __getstate__(cKDTree self):
        cdef object state
        cdef np.intp_t size
        cdef ckdtree * cself = self.cself
        size = cself.tree_buffer.size() * sizeof(ckdtreenode)

        cdef np.ndarray tree = np.asarray(<char[:size]> <char*> cself.tree_buffer.data())

        state = (tree.copy(), self.data.copy(), self.n, self.m, self.leafsize,
                      self.maxes, self.mins, self.indices.copy(),
                      self.boxsize, self.boxsize_data)
        return state

    def __setstate__(cKDTree self, state):
        cdef np.ndarray tree
        cdef ckdtree * cself = self.cself
        cdef np.ndarray mytree

        # unpack the state
        (tree, self.data, self.cself.n, self.cself.m, self.cself.leafsize,
            self.maxes, self.mins, self.indices, self.boxsize, self.boxsize_data) = state

        cself.tree_buffer = new vector[ckdtreenode]()
        cself.tree_buffer.resize(tree.size // sizeof(ckdtreenode))

        mytree = np.asarray(<char[:tree.size]> <char*> cself.tree_buffer.data())

        # set raw pointers
        self._pre_init()

        # copy the tree data
        mytree[:] = tree


        # set up the tree structure pointers
        self._post_init()


def _run_threads(_thread_func, n, n_jobs):
    if n_jobs > 1:
        ranges = [(j * n // n_jobs, (j + 1) * n // n_jobs)
                        for j in range(n_jobs)]

        threads = [threading.Thread(target=_thread_func,
                   args=(start, end))
                   for start, end in ranges]
        for t in threads:
            t.daemon = True
            t.start()
        for t in threads:
            t.join()

    else:
        _thread_func(0, n)
