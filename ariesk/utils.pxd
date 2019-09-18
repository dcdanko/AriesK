
cimport numpy as npc


cdef long [:] convert_kmer(str kmer, int k)
cdef str reverse_convert_kmer(long [:] kmer)

cdef double needle_dist(k1, k2, normalize)
cdef double hamming_dist(k1, k2, normalize)