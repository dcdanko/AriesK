import numpy as np
import sqlite3
cimport numpy as npc


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

    def _bulk_pre_add_kmers(self, lines, sep=','):
        kmers = [lines.strip().split(sep)[0] for line in lines]
        out = [(self._pre_add_kmer(kmer), kmer) for kmer in kmers]
        return out

    cdef str _pre_add_kmer(self, kmer):
        cdef double [:] centroid_rft = np.floor(self.ramifier.c_ramify(kmer) / self.db.box_side_len)
        cdef str centroid_str = ','.join([str(el) for el in centroid_rft])
        return centroid_str

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
