
cimport numpy as npc


cdef npc.uint8_t[:] encode_kmer(str kmer)
cdef str decode_kmer(const npc.uint8_t[:] binary_kmer)

cdef double needle_dist(npc.uint8_t[:] k1, npc.uint8_t[:] k2, bint normalize)
cdef double needle_fast(npc.uint8_t[:] k1, npc.uint8_t[:] k2, bint normalize, double[:, :] score)
cdef double hamming_dist(npc.uint8_t[:] k1, npc.uint8_t[:] k2, bint normalize)
