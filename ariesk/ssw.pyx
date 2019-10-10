# cython: profile=True
# cython: linetrace=True
# cython: language_level=3

# -----------------------------------------------------------------------------
#  Copyright (c) 2013--, scikit-bio development team.
#
#  Distributed under the terms of the Modified BSD License.
#
#  The full license is in the file COPYING.txt, distributed with this software.
# -----------------------------------------------------------------------------

from cpython cimport bool
import numpy as np
cimport numpy as cnp
#from ariesk.lib_ssw cimport *


cdef extern from "_lib/ssw.c":

    ctypedef struct s_align:
        cnp.uint16_t score1
        cnp.uint16_t score2

    ctypedef struct s_profile:
        pass

    cdef s_profile* ssw_init(const cnp.int8_t* read,
                             const cnp.int32_t readLen,
                             const cnp.int8_t* mat,
                             const cnp.int32_t n,
                             const cnp.int8_t score_size)

    cdef void init_destroy(s_profile* p)

    cdef s_align* ssw_align(const s_profile* prof,
                            const cnp.int8_t* ref,
                            cnp.int32_t refLen,
                            const cnp.uint8_t weight_gapO,
                            const cnp.uint8_t weight_gapE,
                            const cnp.uint8_t flag,
                            const cnp.uint16_t filters,
                            const cnp.int32_t filterd,
                            const cnp.int32_t maskLen)


cdef class StripedSmithWaterman:

    def __cinit__(self, cnp.uint8_t[:] query_sequence,
                  gap_open_penalty=5,  # BLASTN Default
                  gap_extend_penalty=2,  # BLASTN Default
                  score_size=2,  # BLASTN Default
                  mask_length=15,  # Minimum length for a suboptimal alignment
                  mask_auto=True,
                  score_only=False,
                  score_filter=None,
                  distance_filter=None,
                  override_skip_babp=False,
                  match_score=2,  # BLASTN Default
                  mismatch_score=-3,  # BLASTN Default
                  ):
        self.gap_open_penalty = gap_open_penalty
        self.gap_extend_penalty = gap_extend_penalty
        self.distance_filter = 0 if distance_filter is None else distance_filter
        self.score_filter = 0 if score_filter is None else score_filter
        self.query_sequence = query_sequence
        self.matrix = self._build_match_matrix(match_score, mismatch_score)
        self.bit_flag = self._get_bit_flag(override_skip_babp, score_only)
        self.mask_length = mask_length
        if mask_auto and (len(query_sequence) // 2) > mask_length:
            self.mask_length = len(query_sequence) // 2

        cdef const cnp.int8_t* query_pointer = <const cnp.int8_t *> &self.query_sequence[0]
        cdef const cnp.int8_t* matrix_pointer = &self.matrix[0]
        self.profile = ssw_init(
            query_pointer,
            query_sequence.shape[0],
            matrix_pointer,
            5,
            score_size
        )


    cdef double align(self, cnp.uint8_t[:] target_sequence):
        self.target_sequence = target_sequence
        cdef s_align *align
        cdef const cnp.int8_t* target_pointer = <const cnp.int8_t *> &self.target_sequence[0]
        align = ssw_align(self.profile, target_pointer,
                          target_sequence.shape[0], self.gap_open_penalty,
                          self.gap_extend_penalty, self.bit_flag,
                          self.score_filter, self.distance_filter,
                          self.mask_length)
        return align.score1

    def __dealloc__(self):
        pass
        if self.profile is not NULL:
            init_destroy(self.profile)


    def _get_bit_flag(self, override_skip_babp, score_only):
        bit_flag = 0
        if score_only:
            return bit_flag
        if override_skip_babp:
            bit_flag = bit_flag | 0x8
        if self.distance_filter != 0:
            bit_flag = bit_flag | 0x4
        if self.score_filter != 0:
            bit_flag = bit_flag | 0x2
        if bit_flag == 0 or bit_flag == 8:
            bit_flag = bit_flag | 0x1
        return bit_flag


    cdef cnp.int8_t[:] _build_match_matrix(self, match_score, mismatch_score):
        sequence_order = "ACGTN"
        dict2d = {}
        for row in sequence_order:
            dict2d[row] = {}
            for column in sequence_order:
                if column == 'N' or row == 'N':
                    dict2d[row][column] = 0
                else:
                    dict2d[row][column] = match_score if row == column \
                        else mismatch_score
        return self._convert_dict2d_to_matrix(dict2d)

    cdef cnp.int8_t[:] _convert_dict2d_to_matrix(self, dict2d):
        sequence_order = "ACGTN"
        cdef int i = 0
        length = len(sequence_order)
        cdef cnp.int8_t[:] py_list_matrix = np.empty(length*length, dtype=np.int8)
        for row in sequence_order:
            for column in sequence_order:
                py_list_matrix[i] = dict2d[row][column]
                i += 1
        return py_list_matrix
