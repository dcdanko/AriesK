
import numpy as np
cimport numpy as npc

from .utils cimport KmerAddable
from .ramft import build_rs_matrix

from json import loads


cdef class Ramifier:
    """Project k-mers into RFT space."""

    def __cinit__(self, k):
        self.k = k
        self.rs_matrix = build_rs_matrix(self.k)

    cdef npc.ndarray c_ramify(self, str kmer):
        cdef long [:, :] binary_kmer = np.array([
            [1 if base == seqb else 0 for seqb in kmer]
            for base in 'ACGT'
        ]).T
        cdef npc.ndarray rft = abs(np.dot(self.rs_matrix, binary_kmer)).flatten()
        return rft

    def ramify(self, str kmer):
        return self.c_ramify(kmer)


cdef class RotatingRamifier:
    """Project k-mers into RFT space with PCA."""

    def __cinit__(self, k, d, rotation, center, scale):
        self.k = k
        self.d = d
        self.rotation = rotation
        self.center = center
        self.scale = scale
        self.ramifier = Ramifier(self.k)

    cdef npc.ndarray c_ramify(self, str kmer):
        cdef npc.ndarray rft = self.ramifier.c_ramify(kmer)
        cdef npc.ndarray centered_scaled = (rft - self.center) / self.scale
        return np.dot(self.rotation, centered_scaled)[:self.d]

    def ramify(self, str kmer):
        return self.c_ramify(kmer)

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


cdef class StatisticalRam(KmerAddable):
    """Identify center, scale, and rotation on a set of k-mers.

    Easier to pre-compute this stuff.
    """

    def __cinit__(self, k, max_size):
        self.k = k
        self.num_kmers_added = 0
        self.max_size = max_size
        self.ramifier = Ramifier(self.k)
        self.rfts = npc.ndarray((self.max_size, 4 * self.k))

    cpdef add_kmer(self, str kmer):
        assert self.num_kmers_added < self.max_size
        cdef double [:] rft = self.ramifier.c_ramify(kmer)
        self.rfts[self.num_kmers_added] = rft
        self.num_kmers_added += 1

    def get_centers(self):
        return np.mean(self.rfts, axis=0)

    def get_scales(self):
        centered = self.rfts - self.get_centers()
        scales = np.max(abs(centered), axis=0)
        return scales

    def get_rotation(self):
        centered_scaled = (self.rfts - self.get_centers()) / self.get_scales()
        R = np.cov(centered_scaled, rowvar=False)
        evals, evecs = np.linalg.eigh(R)
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:,idx]
        return evecs.T

