
import numpy as np
cimport numpy as npc

from libc.math cimport log, floor, ceil, log2

from ariesk.utils cimport encode_kmer


cdef npc.uint64_t fnva(npc.uint8_t[:] data, npc.uint64_t[:] access_order):
    cdef npc.uint64_t hval = 0xcbf29ce484222325
    cdef int i
    cdef max_int = 2 ** 64
    for i in access_order:
        hval = hval ^ data[i]
        hval = hval * 0x100000001b3 % (max_int)
    return hval


cdef class BloomFilter:

    def __cinit__(self, k, int filter_len, npc.uint64_t[:, :] hashes):
        self.p = -1.0
        self.len_seq = k
        self.n_elements = 0

        self.len_filter = filter_len
        self.bitarray = np.zeros((self.len_filter,), dtype=np.uint8)

        self.n_hashes = hashes.shape[0]
        self.hashes = hashes

    def py_add(self, str seq):
        self.add(encode_kmer(seq))

    def py_contains(self, str seq):
        return self.contains(encode_kmer(seq))

    cdef add(self, npc.uint8_t[:] seq):
        self.n_elements += 1
        cdef int i
        cdef npc.uint64_t hval
        for i in range(self.n_hashes):
            hval = fnva(seq, self.hashes[i, :]) 
            hval = hval % self.len_filter
            self.bitarray[hval] = 1

    cdef bint contains(self, npc.uint8_t[:] seq):
        cdef int hashes_hit = 0
        cdef int i
        cdef npc.uint64_t hval
        for i in range(self.n_hashes):
            hval = fnva(seq, self.hashes[i, :])
            hval = hval % self.len_filter
            hashes_hit += self.bitarray[hval]
        return hashes_hit == self.n_hashes

    cdef bint contains_hvals(self, npc.uint64_t[:] hvals):
        cdef int hashes_hit = 0
        cdef int i
        for i in range(hvals.shape[0]):
            hval = hvals[i]
            hashes_hit += self.bitarray[hval]
        return hashes_hit == hvals.shape[0]

    cpdef int union(self, BloomFilter other):
        '''Note. This estimate is pretty bad at the scale we're using.'''
        cdef double bitunion = 1.  # pseudocount
        cdef int i
        for i in range(self.len_filter):
            if (self.bitarray[i] > 0) or (other.bitarray[i] > 0):
                bitunion += 1
        cdef int size_union = <int> ceil(
            (-self.len_filter / self.n_hashes) * log(1 - (bitunion / self.len_filter))
        )
        return size_union

    cpdef int intersection(self, BloomFilter other):
        return self.n_elements + other.n_elements - self.union(other)

    @classmethod
    def build_from_probs(cls, k, expected_size, desired_probability):
        len_filter = expected_size * ceil(-1.44 * log2(desired_probability))
        n_hashes = int(ceil(-log2(desired_probability)))
        hashes = npc.ndarray((n_hashes, k), dtype=np.uint64)
        for i in range(n_hashes):
            for j, val in enumerate(np.random.permutation(k)):
                hashes[i, j] = val
        return cls(k, len_filter, hashes)
