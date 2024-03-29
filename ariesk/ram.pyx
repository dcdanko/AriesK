# cython: profile=True
# cython: linetrace=True
# cython: language_level=3
# cython: boundscheck=False, wraparound=False, nonecheck=False

import numpy as np
cimport numpy as npc
cimport cython
from libc.stdio cimport *
from posix.stdio cimport * # FILE, fopen, fclose
from libc.stdlib cimport malloc, free, rand

from ariesk.utils.ramft import build_rs_matrix

from json import loads

from ariesk.utils.kmers cimport (
    encode_kmer,
    encode_kmer_from_buffer,
    decode_kmer,
)


cdef class Ramifier:
    """Project k-mers into RFT space."""

    def __cinit__(self, k, use_rc=True):
        self.k = k
        self.rs_matrix = build_rs_matrix(self.k)
        self.kmer_matrix = np.zeros((self.k, 4), dtype=np.uint8)
        self.use_rc = use_rc

    cdef npc.ndarray c_ramify(self, npc.uint8_t [::] binary_kmer):
        cdef int i, j, k
        for i in range(self.k):
            for j in range(4):
                self.kmer_matrix[i, j] = 0
            k = binary_kmer[i]
            if k <= 3:  # leave 'N' blank
                self.kmer_matrix[i, k] = 1
            if self.use_rc:
                k = binary_kmer[self.k - i - 1]
                if k <= 3:  # leave 'N' blank
                    k = 3 - k
                    self.kmer_matrix[i, k] = 1
        cdef npc.ndarray rft = np.dot(self.rs_matrix, self.kmer_matrix).flatten()
        return rft

    def ramify(self, str kmer):
        return self.c_ramify(encode_kmer(kmer))


cdef class RotatingRamifier:
    """Project k-mers into RFT space with PCA."""

    def __cinit__(self, k, d, rotation, center, scale, use_scale=True, use_rc=True):
        self.k = k
        self.d = d
        self.rotation = rotation
        self.d_rotation = rotation[:self.d, :]
        self.center = center
        self.scale = scale
        self.ramifier = Ramifier(self.k, use_rc=use_rc)
        self.use_scale = use_scale

    cdef npc.ndarray c_ramify(self, npc.uint8_t [::] binary_kmer):
        cdef npc.ndarray rft = self.ramifier.c_ramify(binary_kmer)
        cdef npc.ndarray centered = (rft - self.center)
        if self.use_scale:
            centered /= self.scale
        return np.dot(self.d_rotation, centered)

    def ramify(self, str kmer):
        return self.c_ramify(encode_kmer(kmer))

    @classmethod
    def from_file(cls, d, filepath):
        saved_rotation = loads(open(filepath).read())
        return cls(
            saved_rotation['k'],
            d,
            np.array(saved_rotation['rotation'], dtype=float),
            np.array(saved_rotation['center'], dtype=float),
            np.array(saved_rotation['scale'], dtype=float),
        )

    @classmethod
    def from_dict(cls, saved_dict):
        return cls(
            saved_dict['k'],
            saved_dict['d'],
            np.array(saved_dict['rotation'], dtype=float),
            np.array(saved_dict['center'], dtype=float),
            np.array(saved_dict['scale'], dtype=float),
        )


cdef class StatisticalRam:
    """Identify center, scale, and rotation on a set of k-mers.

    Easier to pre-compute this stuff.
    """

    def __cinit__(self, k, max_size, use_rc=True):
        self.k = k
        self.num_kmers_added = 0
        self.max_size = max_size
        self.ramifier = Ramifier(self.k, use_rc=use_rc)
        self.rfts = npc.ndarray((self.max_size, 4 * self.k))
        self.closed = False

    def close(self):
        if not self.closed:
            self.rfts = np.array(self.rfts[0:self.num_kmers_added, :])
            self.closed = True

    cpdef add_kmer(self, str kmer):
        assert self.num_kmers_added < self.max_size
        cdef double [:] rft = self.ramifier.c_ramify(encode_kmer(kmer))
        self.rfts[self.num_kmers_added] = rft
        self.num_kmers_added += 1

    cdef c_add_kmer(self, npc.uint8_t [:] kmer):
        cdef double [:] rft = self.ramifier.c_ramify(kmer)
        self.rfts[self.num_kmers_added] = rft
        self.num_kmers_added += 1

    def get_centers(self):
        self.close()
        return np.mean(self.rfts, axis=0)

    def get_scales(self):
        self.close()
        centered = self.rfts - self.get_centers()
        scales = np.std(centered, axis=0)
        return scales

    def get_rotation(self):
        self.close()
        centered_scaled = self.rfts - self.get_centers()
        centered_scaled /= self.get_scales()
        R = np.cov(centered_scaled, rowvar=False)
        evals, evecs = np.linalg.eigh(R)
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:, idx]
        return evecs.T

    def bulk_add_kmers(self, kmers):
        for kmer in kmers:
            self.add_kmer(kmer)

    def add_kmers_from_file(self, str filename, sep=',', start=0, num_to_add=0, preload=False):
        with open(filename) as f:
            n_added = 0
            if start > 0:
                f.readlines(start)
            for line in f:
                if (num_to_add <= 0) or (n_added < num_to_add):
                    kmer = line.split(sep)[0]
                    self.add_kmer(kmer)
                    n_added += 1
            return n_added

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def fast_add_kmers_from_fasta(self, str filename, int dropout=1000):
        cdef FILE * cfile = fopen(filename.encode("UTF-8"), "rb")
        if cfile == NULL:
            raise FileNotFoundError(2, "No such file or directory: '%s'" % filename)

        cdef int n_added = 0
        cdef char * line = NULL
        cdef size_t l = 0
        cdef ssize_t read
        cdef size_t n_kmers_in_line, i
        cdef npc.uint8_t[:] kmer
        while n_added < self.max_size:
            getline(&line, &l, cfile)  # header
            read = getdelim(&line, &l, b'>', cfile)  # read
            if read == -1: break
            while n_added < self.max_size:
                if line[0] == b'\n':
                    line += 1
                kmer = encode_kmer_from_buffer(line, self.ramifier.k)
                if n_added >= self.max_size:
                    break
                if kmer[self.ramifier.k - 1] > 3:
                    break
                if (dropout <= 0) or ((rand() % (1000 * 1000)) < dropout):
                    self.c_add_kmer(kmer)
                    n_added += 1
                line += 1
            line = NULL  # I don't understand why this line is necessary but
                         # without it the program throws a strange error: 
                         # `pointer being realloacted was not allocated`
        fclose(cfile)
        return n_added
