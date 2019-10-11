# cython: language_level=3

import numpy as np
cimport numpy as npc


cdef npc.uint64_t[:, :] seed_and_extend(
    npc.uint8_t[:] query, npc.uint8_t[:] target,
    npc.uint64_t[:, :] q_kmers, npc.uint32_t[:, :] t_kmers,
    int word_size
    )
cdef npc.uint64_t[:, :] extend_seeds(
    npc.uint8_t[:] query, npc.uint8_t[:] target,
    int word_size, int max_intra_interval_gap, int max_inter_interval_gap,
    npc.uint64_t[:, :] matched_positions
    )

cdef npc.uint64_t[:, :] find_matched_positions(npc.uint64_t[:, :] q_kmers, npc.uint32_t[:, :] t_kmers)
cdef npc.uint64_t[:, :] find_compact_intervals(
    int word_size, int max_gap, npc.uint64_t[:, :] matched_positions
    )
cdef int extend_intervals(
    npc.uint8_t[:] query, npc.uint8_t[:] target,
    int max_inter_interval_gap,
    npc.uint64_t[:, :] matching_intervals
    )
cdef extend_seeds_left_right(
    int max_inter_interval_gap,
    npc.uint64_t[:, :] matching_intervals
    )

cdef npc.uint32_t fast_modulo(npc.uint32_t val, npc.uint64_t N)
cdef npc.uint64_t[:, :] get_query_kmers(npc.uint8_t[:] query, int k, int gap)
cdef npc.uint32_t[:, :] get_target_kmers(npc.uint8_t[:] target, int k)
