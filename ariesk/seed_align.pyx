# cython: profile=True
# cython: linetrace=True
# cython: language_level=3

import numpy as np
cimport numpy as npc

from libc.math cimport log, floor, ceil, log2
from ariesk.utils.kmers cimport encode_kmer, decode_kmer, bounded_needle


cdef npc.uint32_t GAP_PENALTY = 1
cdef npc.uint32_t MIS_PENALTY = 1
cdef npc.uint32_t MAX_KMER_HASH = 4096
cdef npc.uint32_t TARGET_WIDTH = 100
cdef npc.uint64_t MAX_INT_32 = 2 ** 32

cdef npc.uint32_t MAX_INTRA_INTERVAL_GAP = 6
cdef npc.uint32_t MAX_INTER_INTERVAL_GAP = 100


def py_seed_extend(query, target, k=7, gap=3):
    return np.array(seed_and_extend(
        encode_kmer(query), encode_kmer(target),
        get_query_kmers(encode_kmer(query), k, gap),
        get_target_kmers(encode_kmer(target), k),
        k
    ))


cdef npc.uint64_t[:, :] seed_and_extend(
    npc.uint8_t[:] query, npc.uint8_t[:] target,
    npc.uint64_t[:, :] q_kmers, npc.uint32_t[:, :] t_kmers,
    int word_size
    ):
    cdef npc.uint64_t[:, :] matched_positions = find_matched_positions(q_kmers, t_kmers)
    return extend_seeds(
        query, target, word_size, MAX_INTRA_INTERVAL_GAP, MAX_INTER_INTERVAL_GAP, matched_positions
    )


cdef npc.uint64_t[:, :] extend_seeds(
    npc.uint8_t[:] query, npc.uint8_t[:] target,
    int word_size, int max_intra_interval_gap, int max_inter_interval_gap,
    npc.uint64_t[:, :] matched_positions
    ):
    """
    Matched positions is a two column list of start positions for words in query
    also found in target. Assume at most one hit per query position. Assume hit
    in target was lowest possible position. Assume targets are monotonically
    not decreasing

    rows are [query_pos, target_pos].

    """
    cdef npc.uint64_t[:, :] matching_intervals = find_compact_intervals(
        word_size, max_intra_interval_gap, matched_positions
    )
    cdef int num_extended = extend_intervals(
        query, target, max_inter_interval_gap, matching_intervals
    )
    matching_intervals = matching_intervals[:num_extended, :]
    # extend_seeds_left_right(max_inter_interval_gap, matching_intervals)

    return matching_intervals


cdef npc.uint64_t[:, :] find_matched_positions(npc.uint64_t[:, :] q_kmers, npc.uint32_t[:, :] t_kmers):
    """Return a two column list of start positions for words in query
    also found in target. At most one hit per query position. Hit in
    target shpould be lowest possible position. Target poss should be
    montotonically not decreasing.

    q_kmers is a two column matrix of [kmer_hash, position]
    t_kmers is a many column matrix with <MAX_KMER_HASH> rows.
        each row contains positions where a single kmer_hash
        occurs in the target. -1 markes the end of the list
    """
    cdef npc.uint64_t[:, :] matched_positions = np.ndarray((q_kmers.shape[0], 2), dtype=np.uint64)
    cdef int n_hits = 0
    cdef int q_index
    cdef int t_index = 0
    cdef int i
    for q_index in range(q_kmers.shape[0]):
        q_kmer_hash = q_kmers[q_index, 0]
        for i in range(t_kmers[q_kmer_hash].shape[0]):
            target_pos = t_kmers[q_kmer_hash, i]
            if target_pos <= 0:
                break
            elif target_pos >= t_index:
                foo = q_kmers[q_index, 1]
                matched_positions[n_hits, 0] = foo
                matched_positions[n_hits, 1] = target_pos
                t_index = target_pos
                n_hits += 1
                break
    return matched_positions


cdef npc.uint64_t[:, :] find_compact_intervals(
    int word_size, int max_gap, npc.uint64_t[:, :] matched_positions
    ):
    """Return a 4 column dataframe with exact intervals that are *almost* exact matches."""
    n_matched_intervals = 0
    cdef npc.uint64_t[:, :] matching_intervals = npc.ndarray((matched_positions.shape[0], 4), dtype=np.uint64)
    cdef int i = 1
    cdef npc.uint64_t current_interval_start_query = matched_positions[0, 0]
    cdef npc.uint64_t current_interval_start_target = matched_positions[0, 1]
    cdef npc.uint64_t current_interval_end_query = matched_positions[0, 0] + word_size - 1
    cdef npc.uint64_t current_interval_end_target = matched_positions[0, 1] + word_size - 1

    while i < matched_positions.shape[0]:
        query_continues = matched_positions[i, 0] <= (current_interval_end_query + max_gap)
        target_continues = matched_positions[i, 1] <= (current_interval_end_target + max_gap)
        if query_continues and target_continues:
            current_interval_end_query = matched_positions[i, 0] + word_size - 1
            current_interval_end_target = matched_positions[i, 1] + word_size - 1
        else:
            matching_intervals[n_matched_intervals, 0] = current_interval_start_query
            matching_intervals[n_matched_intervals, 1] = current_interval_end_query
            matching_intervals[n_matched_intervals, 2] = current_interval_start_target
            matching_intervals[n_matched_intervals, 3] = current_interval_end_target
            n_matched_intervals += 1
            current_interval_start_query = matching_intervals[n_matched_intervals, 0]
            current_interval_start_target = matching_intervals[n_matched_intervals, 1]
            current_interval_end_query = matching_intervals[n_matched_intervals, 0] + word_size - 1
            current_interval_end_target = matching_intervals[n_matched_intervals, 1] + word_size - 1
        i += 1

    matching_intervals[n_matched_intervals, 0] = current_interval_start_query
    matching_intervals[n_matched_intervals, 1] = current_interval_end_query
    matching_intervals[n_matched_intervals, 2] = current_interval_start_target
    matching_intervals[n_matched_intervals, 3] = current_interval_end_target
    n_matched_intervals += 1
    return matching_intervals[:n_matched_intervals, :]


cdef int extend_intervals(
    npc.uint8_t[:] query, npc.uint8_t[:] target,
    int max_inter_interval_gap,
    npc.uint64_t[:, :] matching_intervals
    ):
    """Return a four column dataframe with intervals that have potential
    changes between exact seeds.
    """
    cdef int i = 0
    cdef int j = 1
    while j < matching_intervals.shape[0]:
        q_s, q_e, t_s, t_e = matching_intervals[i, :]
        q_l, t_l = q_e - q_s, t_e - t_s
        n_q_s, n_t_s, _, _ = matching_intervals[i + 1, :]

        gap_score = min(q_l, t_l) - GAP_PENALTY * abs(q_l - t_l)
        q_gap, t_gap = n_q_s - q_e, n_t_s - t_s
        max_gap_btwn_intervals_score = GAP_PENALTY * (max(q_gap, t_gap) - min(q_gap, t_gap))
        max_gap_btwn_intervals_score += MIS_PENALTY * min(q_gap, t_gap)
        if max_gap_btwn_intervals_score <= gap_score:  # automatically extend
            gap_score -= max_gap_btwn_intervals_score
        elif max(q_gap, t_gap) < max_inter_interval_gap:  # attempt extension
            gap_score = bounded_needle(
                query[q_e + 1: n_q_s],
                target[t_e + 1: n_t_s],
                max(q_gap, t_gap) - min(q_gap, t_gap)
            )
            gap_score *= -1  # our function returns distance not similarity
        else:
            gap_score = -1
        if gap_score >= 0:  # extension succesful, merge the intervals
            matching_intervals[i, 1] = matching_intervals[j, 1]
            matching_intervals[i, 3] = matching_intervals[j, 3]
        else:  # extension failed, try from the next interval
            i = j
        j += 1  # test if we can extend to the next untested interval)
    return i + 1

cdef extend_seeds_left_right(
    int max_inter_interval_gap,
    npc.uint64_t[:, :] matching_intervals
    ):
    """Extend the intervals left and right, very heuristic."""
    cdef int i
    for i in range(matching_intervals.shape[0]):
        q_s, q_e, t_s, t_e = matching_intervals[i, :]
        max_extend = min(q_e - q_s, t_e - t_s) // 4  # extend left and right for a bit
        max_extend = min(max_extend, max_inter_interval_gap)
        q_i, t_i = q_e + 1, t_e + 1
        max_extend_left = max_extend
        if (q_s - max_extend_left) < 0 or (t_s - max_extend_left) < 0:
            max_extend_left = min(q_s, t_s)
        matching_intervals[i, 0] = q_s - max_extend_left
        matching_intervals[i, 1] = q_e + max_extend
        matching_intervals[i, 2] = t_s - max_extend_left
        matching_intervals[i, 3] = t_e + max_extend


'''
k-mer hash functions
'''

cdef npc.uint32_t fast_modulo(npc.uint32_t val, npc.uint64_t N):
    """Technically not a modulo but serve the same purpose faster.

    N.B. This is ONLY useful if val is pseudo-random. For sequential
    numbers it will likely produce the same value. 
    """
    cdef npc.uint64_t val64 = <npc.uint64_t> val
    cdef npc.uint32_t shifted = (val * N) >> 32
    return shifted


cdef npc.uint32_t fnva(npc.uint8_t[:] data):
    cdef npc.uint32_t hval = 0x811c9dc5
    cdef int i
    for i in range(data.shape[0]):
        hval = hval ^ data[i]
        hval = fast_modulo(hval * 0x01000193, MAX_INT_32)
    return hval


cdef npc.uint64_t[:, :] get_query_kmers(npc.uint8_t[:] query, int k, int gap):
    cdef npc.uint64_t[:, :] q_kmers = np.ndarray(
        ((query.shape[0] - k + 1) // gap, 2),
        dtype=np.uint64
    )
    cdef int n_kmers = 0
    cdef int i
    for i in range(0, query.shape[0] - k + 1, gap):
        if i + k < query.shape[0] and n_kmers < q_kmers.shape[0]:
            kmer_hash = fast_modulo(fnva(query[i:i + k]), MAX_KMER_HASH)
            q_kmers[n_kmers, 0] = kmer_hash
            q_kmers[n_kmers, 1] = i
            n_kmers += 1
    print(np.array(q_kmers[:n_kmers,:]))
    return q_kmers[:n_kmers,:]


cdef npc.uint32_t[:, :] get_target_kmers(npc.uint8_t[:] target, int k):
    cdef npc.uint32_t[:, :] t_kmers = np.zeros((MAX_KMER_HASH, TARGET_WIDTH), dtype=np.uint32)
    cdef int i, j
    cdef npc.uint32_t kmer_hash
    for i in range(1, target.shape[0] - k + 1):
        kmer_hash = fast_modulo(fnva(target[i:i + k]), MAX_KMER_HASH)
        for j in range(TARGET_WIDTH):
            if t_kmers[kmer_hash, j] <= 0:
                t_kmers[kmer_hash, j] = i
                break
    return t_kmers
