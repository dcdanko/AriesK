
import numpy as np
cimport numpy as npc


cdef npc.uint8_t [::] encode_kmer(str kmer):
    cdef dict base_map = {'A': 0., 'C': 1., 'G': 2, 'T': 3}
    cdef npc.uint8_t [::] binary_kmer = np.array(
        [base_map[base] for base in kmer], dtype=np.uint8
    )
    return binary_kmer


cdef npc.uint8_t [::] encode_kmer_from_buffer(char * buf, int k):
    cdef npc.uint8_t[::] kmer = np.ndarray((k,), dtype=np.uint8)
    kmer[k - 1] = 255  # we use this as a code to indicate the kmer was not fully read
    cdef int i = 0
    while i < k:
        c = buf[0]
        if c == 0:
            break  # this means the buffer did not have enough to read
        if c == 'A':
            kmer[i] = 0
        elif c == 'C':
            kmer[i] = 1
        elif c == 'G':
            kmer[i] = 2
        elif c == 'T':
            kmer[i] = 3
        elif c == '\n':
            i -= 1  # special case for line wrapping in fasta
        i += 1
        buf += 1
    return kmer



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


def py_needle(kmers, normalize=False):
    out = []
    for i, k1 in enumerate(kmers):
        for j, k2 in enumerate(kmers):
            if i < j:
                out.append((k1, k2, needle_dist(encode_kmer(k1), encode_kmer(k2), normalize)))
    return out


def py_needle_fast(kmers, normalize=False):
    out = []
    cdef double[:, :] score = np.zeros((len(kmers[0]) + 1, len(kmers[0]) + 1))
    for i, k1 in enumerate(kmers):
        for j, k2 in enumerate(kmers):
            if i < j:
                out.append((k1, k2, needle_fast(
                    encode_kmer(k1), encode_kmer(k2), normalize, score
                )))
    return out


cdef double needle_dist(npc.uint8_t[::] k1, npc.uint8_t[::] k2, bint normalize):
    """Return the NW alignment distance."""
    cdef double[:, :] score = np.zeros((k1.shape[0] + 1, k2.shape[0] + 1))
    return needle_fast(k1, k2, normalize, score)


cdef double needle_fast(npc.uint8_t[::] k1, npc.uint8_t[::] k2, bint normalize, double[:, :] score):
    """Return NW alignment using pre-allocated RAM."""
    cdef double match_score = 0
    cdef double mismatch_penalty = 1
    cdef double gap_penalty = 1
    cdef int i, j
    for i in range(k1.shape[0] + 1):
        score[i][0] = gap_penalty * i
    for j in range(k2.shape[0] + 1):
        score[0][j] = gap_penalty * j

    for i in range(1, k1.shape[0] + 1):
        for j in range(1, k2.shape[0] + 1):
            if k1[i - 1] == k2[j - 1]:
                match = score[i - 1][j - 1]
            else:
                match = score[i - 1][j - 1] + mismatch_penalty
            delete = score[i - 1][j] + gap_penalty
            insert = score[i][j - 1] + gap_penalty
            score[i][j] = min(match, delete, insert)
    final_score = score[k1.shape[0]][k2.shape[0]]
    if normalize:
        final_score /= k1.shape[0]
    return final_score


cdef double bounded_needle_fast(npc.uint8_t[::] k1, npc.uint8_t[::] k2, npc.uint8_t bound, bint normalize, double[:, :] score):
    """Return NW alignment using pre-allocated RAM."""
    cdef double match_score = 0
    cdef double mismatch_penalty = 1
    cdef double gap_penalty = 1
    cdef int i, j, o
    for i in range(k1.shape[0] + 1):
        for o in range(bound + 1):
            j = i + o
            if j < (k1.shape[0] + 1):
                score[i][j] = gap_penalty * o
                score[j][i] = gap_penalty * o

    for i in range(1, k1.shape[0] + 1):
        for o in range(-bound, bound + 1):
            j = i + o
            if j < (k1.shape[0] + 1):
                if k1[i - 1] == k2[j - 1]:
                    match = score[i - 1][j - 1]
                else:
                    match = score[i - 1][j - 1] + mismatch_penalty
                delete = score[i - 1][j] + gap_penalty
                insert = score[i][j - 1] + gap_penalty
                score[i][j] = min(match, delete, insert)
    final_score = score[k1.shape[0]][k2.shape[0]]
    if normalize:
        final_score /= k1.shape[0]
    return final_score