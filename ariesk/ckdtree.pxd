
import numpy as np
cimport numpy as np
from libcpp.vector cimport vector


cdef extern from "<limits.h>":
    long LONG_MAX


cdef extern from *:
    int NPY_LIKELY(int)
    int NPY_UNLIKELY(int)


cdef extern from "_lib/kdtree/ckdtree.cpp":

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
    cdef:
        ckdtree * cself
        readonly double[:, :]    data
        readonly np.ndarray      maxes
        readonly np.ndarray      mins
        readonly np.ndarray      indices
        readonly object          boxsize
        np.ndarray               boxsize_data

    cdef public object logger
    cdef public bint logging

    cdef _pre_init(cKDTree self)
    cdef _post_init(cKDTree self)
    cdef _post_init_traverse(cKDTree self, ckdtreenode *node)
    cdef query_ball_tree(
        cKDTree self, cKDTree other, np.float64_t r,
        np.float64_t p=?,
        np.float64_t eps=?
    )

