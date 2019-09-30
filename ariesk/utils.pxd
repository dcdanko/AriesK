
cimport numpy as npc


cdef npc.uint8_t[:] encode_kmer(str kmer)
cdef npc.uint8_t [:] encode_kmer_from_buffer(char * buf, int k)
cdef str decode_kmer(const npc.uint8_t[:] binary_kmer)

cdef double needle_dist(npc.uint8_t[:] k1, npc.uint8_t[:] k2, bint normalize)
cdef double needle_fast(npc.uint8_t[:] k1, npc.uint8_t[:] k2, bint normalize, double[:, :] score)
cdef double bounded_needle_fast(npc.uint8_t[:] k1, npc.uint8_t[:] k2, npc.uint8_t bound, bint normalize, double[:, :] score)
cdef double hamming_dist(npc.uint8_t[:] k1, npc.uint8_t[:] k2, bint normalize)
