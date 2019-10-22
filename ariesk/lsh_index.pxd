
cimport numpy as npc
import numpy as np


cdef class LSHIndex:
    cdef public dict tbl
    cdef public int dims
    cdef public double[:, :] delta
    
    cdef set query(LSHIndex self, double[:] pt)
