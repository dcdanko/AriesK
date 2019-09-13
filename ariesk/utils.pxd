

cdef long convert_kmer(str kmer)
cdef str reverse_convert_kmer(long kmer)

cdef  class KmerAddable:
    cdef public long num_kmers_added, max_size

    cpdef add_kmer(self, str kmer)