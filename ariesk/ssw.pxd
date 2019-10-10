# cython: language_level=3

from cpython cimport bool
import numpy as np
cimport numpy as cnp
#from ariesk.lib_ssw cimport *


cdef extern from "_lib/ssw.c":

    ctypedef struct s_profile:
        pass



cdef class StripedSmithWaterman:
    cdef s_profile *profile
    cdef cnp.uint8_t gap_open_penalty
    cdef cnp.uint8_t gap_extend_penalty
    cdef cnp.uint8_t bit_flag
    cdef cnp.uint16_t score_filter
    cdef cnp.int32_t distance_filter
    cdef cnp.int32_t mask_length
    cdef int index_starts_at
    cdef bool is_protein
    cdef bool suppress_sequences
    cdef cnp.uint8_t[:] query_sequence
    cdef cnp.uint8_t[:] target_sequence
    cdef cnp.int8_t[:] matrix

    cdef double align(self, cnp.uint8_t[:] target_sequence)
    cdef cnp.int8_t[:] _build_match_matrix(self, match_score, mismatch_score)
    cdef cnp.int8_t[:] _convert_dict2d_to_matrix(self, dict2d)
