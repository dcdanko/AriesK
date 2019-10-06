# cython: profile=True
# cython: linetrace=True
# cython: language_level=3

import numpy as np
cimport numpy as npc

from libc.math cimport log, floor, ceil, log2

from ariesk.utils.kmers cimport encode_kmer, decode_kmer


cdef npc.uint32_t fast_modulo(npc.uint32_t val, npc.uint64_t N, npc.uint32_t shift):
    """Technically not a modulo but serve the same purpose faster."""
    return <npc.uint32_t> ((<npc.uint64_t> val) * N) >> shift


cdef npc.uint32_t fnva(npc.uint8_t[:] data, npc.uint64_t[:] access_order):
    cdef npc.uint32_t hval = 0xcbf29ce484222325
    cdef int i
    cdef npc.uint64_t max_int = 2 ** 32
    for i in access_order:
        hval = hval ^ data[i]
        hval = fast_modulo(hval * 0x100000001b3, max_int, 32)
    return hval


cdef class BloomFilter:

    def __cinit__(self, k, int filter_len, npc.uint64_t[:, :] hashes):
        self.p = -1.0
        self.len_seq = k
        self.n_elements = 0
        for i in range(100):
            if (2 ** i) >= filter_len:
                self.len_filter = 2 ** i
                self.filter_power = i
                break
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
        cdef npc.uint32_t hval
        for i in range(self.n_hashes):
            hval = fnva(seq, self.hashes[i, :])
            hval = fast_modulo(hval, self.len_filter, self.filter_power)
            self.bitarray[hval] = 1

    cdef bint contains(self, npc.uint8_t[:] seq):
        cdef int hashes_hit = 0
        cdef int i
        cdef npc.uint32_t hval
        for i in range(self.n_hashes):
            hval = fnva(seq, self.hashes[i, :])
            hval = fast_modulo(hval, self.len_filter, self.filter_power)
            hashes_hit += self.bitarray[hval]
        return hashes_hit == self.n_hashes

    cdef bint contains_hvals(self, npc.uint64_t[:] hvals):
        cdef int hashes_hit = 0
        cdef int i
        for i in range(hvals.shape[0]):
            hashes_hit += self.bitarray[hvals[i]]
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


cdef class BloomGrid:

    def __cinit__(self, int col_k, int row_k, int grid_width, int grid_height,
                  npc.uint64_t[:, :] row_hashes, npc.uint64_t[:, :] col_hashes):
        for i in range(100):
            if (2 ** i) >= grid_width:
                self.grid_width = 2 ** i
                self.grid_width_power = i
                break
        self.grid_height = grid_height
        self.col_k = col_k
        self.row_k = row_k
        self.bitarray = np.zeros((self.grid_width,), dtype=np.uint8)
        self.bitgrid = np.zeros((self.grid_height, self.grid_width), dtype=np.uint8)
        self.col_hashes = col_hashes
        self.row_hashes = row_hashes

    def py_add(self, str seq):
        """Add a full length seq to the BloomGrid."""
        assert len(seq) == self.row_k
        self.add(encode_kmer(seq))

    cdef add(self, npc.uint8_t[:] seq):
        """Add a full length seq to the BloomGrid."""
        cdef int i, j
        cdef npc.uint64_t col_hval, row_hval
        cdef npc.uint64_t[:] row_hvals = npc.ndarray((self.row_hashes.shape[0],), dtype=np.uint64)
        for j in range(self.row_hashes.shape[0]):
            row_hvals[j] = fnva(seq, self.row_hashes[j, :]) % self.grid_height
        for i in range(self.row_k - self.col_k + 1):
            for j in range(self.col_hashes.shape[0]):
                col_hval = fast_modulo(
                    fnva(seq[i:i + self.col_k], self.col_hashes[j, :]),
                    self.grid_width,
                    self.grid_width_power
                )
                self.bitarray[col_hval] = 1
                for j in range(self.row_hashes.shape[0]):
                    self.bitgrid[row_hvals[j], col_hval] = 1

    cdef npc.uint64_t[:] _get_hashes(self, npc.uint8_t[:] seq):
        cdef int i
        cdef npc.uint64_t[:] hash_vals = np.ndarray((self.col_hashes.shape[0],), dtype=np.uint64)
        for i in range(self.col_hashes.shape[0]):
            hash_vals[i] = fast_modulo(
                fnva(seq, self.col_hashes[i, :]),
                self.grid_width,
                self.grid_width_power
            )
        return hash_vals

    def py_array_contains(self, str seq):
        return self.array_contains(encode_kmer(seq))     

    cdef bint array_contains(self, npc.uint8_t[:] seq):
        cdef npc.uint64_t[:] hash_vals = self._get_hashes(seq)
        return self.array_contains_hvals(hash_vals)

    cdef bint array_contains_hvals(self, npc.uint64_t[:] hvals):
        assert hvals.shape[0] == self.col_hashes.shape[0]
        cdef int hashes_hit = 0
        cdef int i
        for i in range(hvals.shape[0]):
            hashes_hit += self.bitarray[hvals[i]]
        return hashes_hit == hvals.shape[0]

    def py_grid_contains(self, str seq):
        return self.grid_contains(encode_kmer(seq))

    cdef npc.uint8_t[:] grid_contains(self, npc.uint8_t[:] seq):
        """Return a vector of 1/0 based on whether a row contains a given sub-kmer."""
        cdef npc.uint64_t[:] hash_vals = self._get_hashes(seq)
        return self.grid_contains_hvals(hash_vals)

    cdef npc.uint8_t[:] grid_contains_hvals(self, npc.uint64_t[:] hvals):
        cdef npc.uint8_t[:] rows_hit = np.ndarray((self.grid_height,), dtype=np.uint8)
        cdef int hashes_hit
        cdef int i, j
        for i in range(self.grid_height):
            hashes_hit = 0
            for j in range(hvals.shape[0]):
                hashes_hit += self.bitgrid[i, hvals[j]]
            rows_hit[i] = 1
        return rows_hit

    def py_count_grid_contains(self, str seq):
        return list(self.count_grid_contains(encode_kmer(seq)))

    cdef npc.uint8_t[:] count_grid_contains(self, npc.uint8_t[:] seq):
        """Return a vector counting the number of sub_kmers that hit each row.

        Takes a full length seq as input."""
        cdef npc.uint64_t[:] row_hash_vals
        cdef npc.uint64_t[:, :] hash_vals = np.ndarray(
            (seq.shape[0] - self.col_k + 1, self.col_hashes.shape[0]),
            dtype=np.uint64
        )
        cdef int i, j
        for i in range(seq.shape[0] - self.col_k + 1):
            row_hash_vals = self._get_hashes(seq[i:i + self.col_k])
            for j in range(self.col_hashes.shape[0]):
                hash_vals[i, j] = row_hash_vals[j]
        return self.count_grid_contains_hvals(hash_vals)

    cdef  npc.uint8_t[:] count_grid_contains_hvals(self, npc.uint64_t[:, :] hvals):
        """Return a numpy array with the number of sub-kmers hit per row in grid.

        hvals is a matrix of (sub_kmers, num_hashes)
        """
        cdef npc.uint8_t[:] rows_hit = np.zeros((self.grid_height,), dtype=np.uint8)
        cdef int hashes_hit
        cdef int s, i, j
        for s in range(hvals.shape[0]):  # iterate over sub-kmers
            for i in range(self.grid_height):  # iterate over rows in grid
                hashes_hit = 0
                for j in range(hvals[s].shape[0]):  # iterate over hashes
                    hashes_hit += self.bitgrid[i, hvals[s, j]]
                if hashes_hit == hvals[s].shape[0]:  # check if we got a sub-kmer on this row
                    rows_hit[i] += 1
        return rows_hit

    @classmethod
    def build_from_probs(cls, k, sub_k, grid_height, n_row_hashes, expected_size, desired_probability):
        len_filter = expected_size * ceil(-1.44 * log2(desired_probability))
        n_col_hashes = int(ceil(-log2(desired_probability)))
        col_hashes = npc.ndarray((n_col_hashes, sub_k), dtype=np.uint64)
        for i in range(n_col_hashes):
            for j, val in enumerate(np.random.permutation(sub_k)):
                col_hashes[i, j] = val
        row_hashes = npc.ndarray((n_row_hashes, k), dtype=np.uint64)
        for i in range(n_row_hashes):
            for j, val in enumerate(np.random.permutation(k)):
                row_hashes[i, j] = val
        return cls(sub_k, k, len_filter, grid_height, row_hashes, col_hashes)
