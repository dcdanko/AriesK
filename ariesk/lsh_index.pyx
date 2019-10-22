
from ariesk.dbs.contig_db cimport ContigDB
cimport numpy as npc
import numpy as np


cdef class LSHIndex:
    
    def __cinit__(LSHIndex self, ContigDB db, double[:, :] pts, int dims, int radius):
        self.tbl = {}
        self.dims = dims
        cdef int i, d, r
        cdef tuple key
        for i in range(pts.shape[0]):
            key = tuple(pts[i, :self.dims])
            try:
                self.tbl[key] |= set(db.get_coords(i))
            except KeyError:
                self.tbl[key] = set(db.get_coords(i))

        cdef int alphabet_size = 2 * radius + 1
        self.delta = -9 * np.ones(
            (alphabet_size ** self.dims, self.dims),
            dtype=float
        )
        for d in range(self.dims):
            i = 0
            while i < self.delta.shape[0]:
                for r in range(-radius, radius + 1):
                    for _ in range(alphabet_size ** d):
                        self.delta[i, d] = r
                        i += 1

    cdef set query(LSHIndex self, double[:] pt):
        cdef int i, j
        cdef tuple key
        cdef set out = set()
        cdef double[:] mu = np.ndarray((self.dims,))
        for i in range(self.delta.shape[0]):
            for j in range(self.dims):
                mu[j] = pt[j] + self.delta[i, j]
            key = tuple(mu)
            if key in self.tbl:
                out |= self.tbl[key]
        return out