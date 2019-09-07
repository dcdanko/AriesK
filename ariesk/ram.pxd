
cimport numpy as npc
from .utils cimport KmerAddable


cdef class Ramifier:
    """Project k-mers into RFT space."""
    cdef public long k
    cdef public double [:, :] rs_matrix

    cdef npc.ndarray c_ramify(self, str kmer)


cdef class RotatingRamifier:
    """Project k-mers into RFT space with PCA."""
    cdef public Ramifier ramifier
    cdef public long k, d
    cdef public double [:, :] rs_matrix, rotation
    cdef public double [:] center, scale

    cdef npc.ndarray c_ramify(self, str kmer)


cdef class StatisticalRam(KmerAddable):
    """Identify center, scale, and rotation on a set of k-mers.

    Easier to pre-compute this stuff.
    """
    cdef public long k
    cdef public Ramifier ramifier
    cdef public double [:, :] rfts

