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
    cdef public dict centroid_id_cache
    cdef public dict contig_kmer_cache
    cdef public object logger
    cdef public bint logging

    cpdef _build_tables(self)
    cpdef _build_indices(self)
    cpdef _drop_indices(self)
    cpdef list get_coords(self, int centroid_id)
    cpdef tuple get_contig(self, int seq_coord)
    cdef npc.uint32_t[:, :] get_contig_kmers(self, int seq_coord, int k)
    cdef add_contig_seq(self, str genome_name, str contig_name, int seq_coord, int contig_coord, npc.uint8_t[:] contig_section)
    cdef add_coord_to_centroid(self, int centroid_id, int seq_coord)
    cdef add_contig(self, str genome_name, str contig_name, npc.uint8_t[:] contig, int gap=?)
    cdef _clear_coord_buffer(self)