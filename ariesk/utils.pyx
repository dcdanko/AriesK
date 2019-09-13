

cdef long convert_kmer(str kmer):
    cdef long out = 0
    for i, base in enumerate(kmer):
        val = 0
        if base == 'C':
            val = 1
        elif base == 'G':
            val = 2
        elif base == 'T':
            val = 3
        out += val + (4 ** i) 
    return out


cdef class KmerAddable:

    cpdef add_kmer(self, str kmer):
        raise NotImplementedError()

    def bulk_add_kmers(self, kmers):
        for kmer in kmers:
            self.add_kmer(kmer)

    def add_kmers_from_file(self, str filename, sep=',', start=0):
        with open(filename) as f:
            for i, line in enumerate(f):
                if i < start:
                    continue
                if (i - start) >= self.max_size:
                    break
                kmer = line.split(sep)[0]
                self.add_kmer(kmer)
