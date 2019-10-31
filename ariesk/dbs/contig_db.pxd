# cython: language_level=3

import sqlite3
import numpy as np
cimport numpy as npc

from ariesk.dbs.core_db cimport CoreDB


cdef class ContigDB(CoreDB):
    cdef public int seq_block_len
    cdef public int current_seq_coord
    cdef public set genomes_added
    cdef public int coord_buffer_filled
    cdef public list coord_buffer
    cdef public dict contig_cache
    cdef public dict seq_cache
    cdef public dict centroid_id_cache
    cdef public dict contig_kmer_cache

    cpdef _build_tables(self)
    cpdef _build_indices(self)
    cpdef _drop_indices(self)
    cpdef list get_contigs(self, int centroid_id)
    cdef add_contig_seq(self,
                        str contig_name, int centroid_id,
                        int start_coord, int end_coord)
    cdef add_contig(self, str genome_name, str contig_name, npc.uint8_t[:] contig, int gap=?)
    cdef npc.uint8_t[:] get_seq(self, str contig_name, int start_coord, int end_coord)