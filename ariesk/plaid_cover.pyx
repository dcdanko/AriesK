
import numpy as np
cimport numpy as npc

from .utils cimport convert_kmer, KmerAddable
from .ram cimport RotatingRamifier


cdef class PlaidCoverBuilder(KmerAddable):
    cdef public RotatingRamifier ramifier
    cdef public float box_side_len
    cdef public long [:] kmers
    cdef public double [:, :] rfts
    cdef public object clusters

    def __cinit__(self, box_side_len, max_size, ramifier):
        self.box_side_len = box_side_len
        self.ramifier = ramifier
        self.max_size = max_size
        self.num_kmers_added = 0
        self.kmers = npc.ndarray((self.max_size,), dtype=long)
        self.rfts = npc.ndarray((self.max_size, self.ramifier.d))

        self.clusters = {}

    cpdef add_kmer(self, str kmer):
        assert self.num_kmers_added < self.max_size
        cdef long kmer_code = convert_kmer(kmer)
        self.kmers[self.num_kmers_added] = kmer_code
        cdef double [:] rft = self.ramifier.c_ramify(kmer)
        self.rfts[self.num_kmers_added] = rft
        self.num_kmers_added += 1


    cpdef cluster(self):
        cdef long [:, :] Y = npc.ndarray((self.num_kmers_added, self.ramifier.d), dtype=long)
        cdef long [:] ordered_indices
        for j in range(self.ramifier.d):
            ordered_indices = np.argsort(self.rfts[:, j])
            p, z_n = 0, self.rfts[ordered_indices[-1], j]
            while self.rfts[ordered_indices[p], j] + self.box_side_len < z_n:
                box_end = self.rfts[ordered_indices[p], j] + self.box_side_len
                '''
                Add points to the box (on this axis 'j') until points are no
                longer in the box.
                ''' 
                i = p
                while box_end > self.rfts[ordered_indices[i], j]:
                    Y[ordered_indices[i], j] = p
                    i += 1
                p = i  # 'i' is now the smallest point not in the box
            i = p  # Add remaining points to the final box
            while i < self.num_kmers_added:
                Y[ordered_indices[i], j] = p
                i += 1
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
