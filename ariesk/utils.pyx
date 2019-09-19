
import numpy as np
cimport numpy as npc


cdef npc.uint8_t [:] encode_kmer(str kmer):
    cdef dict base_map = {'A': 0., 'C': 1., 'G': 2, 'T': 3}
    cdef npc.uint8_t [:] binary_kmer = np.array(
        [base_map[base] for base in kmer], dtype=np.uint8
    )
    return binary_kmer


cdef str decode_kmer(const npc.uint8_t [:] binary_kmer):
    cdef dict base_map = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    cdef int i
    cdef str out = ''
    for i in range(binary_kmer.shape[0]):
        out += base_map[binary_kmer[i]]
    return out


def py_encode_kmer(str kmer):
    return np.array(encode_kmer(kmer), dtype=np.uint8)


def py_decode_kmer(npc.ndarray binary_kmer):
    return decode_kmer(binary_kmer)


cdef double hamming_dist(npc.uint8_t [:] k1, npc.uint8_t [:] k2, bint normalize):
    cdef double score = 0
    cdef int i
    for i in range(k1.shape[0]):
        score += 1 if k1[i] != k2[i] else 0
    if normalize:
        score /= k1.shape[0]
    return score


cdef double needle_dist(npc.uint8_t [:] k1, npc.uint8_t [:] k2, bint normalize):
    cdef double [:, :] score = np.zeros((k1.shape[0] + 1, k2.shape[0] + 1))
    cdef double match_score = 0
    cdef double mismatch_penalty = 1
    cdef double gap_penalty = 1
    cdef int i, j
    for i in range(k1.shape[0] + 1):
        score[i][0] = gap_penalty * i
    for j in range(k2.shape[0] + 1):
        score[0][j] = gap_penalty * j

    def _match_score(b1, b2):
        return match_score if b1 == b2 else mismatch_penalty

    for i in range(1, k1.shape[0] + 1):
        for j in range(1, k2.shape[0] + 1):
            match = score[i - 1][j - 1] + _match_score(k1[i - 1], k2[j - 1])
            delete = score[i - 1][j] + gap_penalty
            insert = score[i][j - 1] + gap_penalty
            score[i][j] = min(match, delete, insert)
    final_score = score[k1.shape[0]][k2.shape[0]]
    if normalize:
        final_score /= k1.shape[0]
    return final_score
