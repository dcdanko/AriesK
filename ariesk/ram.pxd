# cython: language_level=3

cimport numpy as npc


cdef class Ramifier:
    """Project k-mers into RFT space."""
    cdef public long k
    cdef public double [:, :] rs_matrix
    cdef npc.uint8_t [:, :] kmer_matrix
    cdef public bint use_rc

    cdef npc.ndarray c_ramify(self, npc.uint8_t [::] binary_kmer)


cdef class RotatingRamifier:
    """Project k-mers into RFT space with PCA."""
    cdef public Ramifier ramifier
    cdef public long k, d
    cdef public bint use_scale
    cdef public double [:, :] rotation
    cdef public double [:, :] d_rotation
    cdef public double [:] center, scale

    cdef npc.ndarray c_ramify(self, npc.uint8_t [::] binary_kmer)

cdef class StatisticalRam:
    """Identify center, scale, and rotation on a set of k-mers.

    Easier to pre-compute this stuff.
    """
    cdef public long k
    cdef public Ramifier ramifier
    cdef public double [:, :] rfts
    cdef public int num_kmers_added
    cdef public int max_size
    cdef public bint closed

    cpdef add_kmer(self, str kmer)
    cdef c_add_kmer(self, npc.uint8_t [:] binary_kmer)
