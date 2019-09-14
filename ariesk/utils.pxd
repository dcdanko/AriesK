
cimport numpy as npc


cdef long [:] convert_kmer(str kmer, int k)
cdef str reverse_convert_kmer(long [:] kmer)

cdef double needle_dist(k1, k2)

cdef  class KmerAddable:
    cdef public long num_kmers_added, max_size

    cpdef add_kmer(self, str kmer)