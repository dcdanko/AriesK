import numpy as np
cimport numpy as npc

from cython.parallel import prange

from .utils cimport convert_kmer, KmerAddable
from .ram cimport RotatingRamifier


cdef class GridCoverBuilder(KmerAddable):
    cdef public RotatingRamifier ramifier
    cdef public float box_side_len
    cdef public long [:, :] kmers
    cdef public double [:, :] rfts
    cdef public object clusters
    cdef public int threads

    def __cinit__(self, box_side_len, max_size, ramifier, threads=1):
        self.box_side_len = box_side_len
        self.ramifier = ramifier
        self.max_size = max_size
        self.num_kmers_added = 0
        self.kmers = npc.ndarray((self.max_size, self.ramifier.k), dtype=long)
        self.rfts = npc.ndarray((self.max_size, self.ramifier.d))
        self.threads = threads

        self.clusters = {}

    cpdef add_kmer(self, str kmer):
        assert self.num_kmers_added < self.max_size
        cdef double [:] rft = self.ramifier.c_ramify(kmer)
        self.kmers[self.num_kmers_added] = convert_kmer(kmer, self.ramifier.k)
        self.rfts[self.num_kmers_added] = rft
        self.num_kmers_added += 1

    cpdef cluster(self):
        cdef long [:, :] Y = npc.ndarray((self.num_kmers_added, self.ramifier.d), dtype=long)
        cdef int i, j
        for j in range(self.ramifier.d):
            for i in range(self.num_kmers_added):
                Y[i, j] = np.floor(self.rfts[i, j] / self.box_side_len)
        '''
        At this point Y contains N unique points where N <= self.num_kmers_added

        The values of each point are the index of the 'centroid' in each
        dimension. Centroid in quotes because each point in Y actually 
        has D centroids which collectively define a box.
        '''
        for i in range(self.num_kmers_added):
            point = tuple(Y[i, :])
            self.clusters[point] = [i] + self.clusters.get(point, [])

    def to_dict(self):
        out = {
            'type': 'grid_cover',
            'box_side_length': self.box_side_len,
            'ramifier': {
                'k': self.ramifier.k,
                'd': self.ramifier.d,
                'center': np.asarray(self.ramifier.center).tolist(),
                'scale': np.asarray(self.ramifier.scale).tolist(),
                'rotation': np.asarray(self.ramifier.rotation).tolist(),
            },
            'kmers': np.asarray(self.kmers).tolist(),
            'clusters': [],
        }
        for centroid, members in self.clusters.items():
            out['clusters'].append({
                'centroid': centroid,
                'members': members,
            })
        return out
