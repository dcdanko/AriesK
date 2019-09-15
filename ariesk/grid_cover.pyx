import numpy as np
import sqlite3
cimport numpy as npc

from cython.parallel import prange

from .utils cimport convert_kmer, KmerAddable
from .ram cimport RotatingRamifier
from .db cimport GridCoverDB


cdef class GridCoverBuilder(KmerAddable):
    cdef public RotatingRamifier ramifier
    cdef public GridCoverDB db

    def __cinit__(self, db):
        self.db = db
        self.ramifier = db.ramifier
        self.num_kmers_added = 0

    cpdef add_kmer(self, str kmer):
        cdef double [:] centroid_rft = np.floor(self.ramifier.c_ramify(kmer) / self.db.box_side_len)
        self.db.add_point_to_cluster(centroid_rft, kmer)
        self.num_kmers_added += 1

    def commit(self):
        self.db.commit()

    def close(self):
        self.db.close()

    @classmethod
    def from_filepath(cls, filepath, ramifier, box_side_len):
        db = GridCoverDB(sqlite3.connect(filepath), ramifier=ramifier, box_side_len=box_side_len)
        return cls(db)

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
