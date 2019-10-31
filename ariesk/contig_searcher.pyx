# cython: profile=True
# cython: linetrace=True
# cython: language_level=3

cimport numpy as npc
import numpy as np
from libc.stdio cimport *
from posix.stdio cimport * # FILE, fopen, fclose
from libc.stdlib cimport malloc, free
from ariesk.ckdtree cimport cKDTree
from ariesk.ssw cimport StripedSmithWaterman
from ariesk.dbs.contig_db cimport ContigDB
from ariesk.utils.kmers cimport (
    needle_dist,
    hamming_dist,
    encode_kmer,
    decode_kmer,
    encode_seq_from_buffer,
)
from ariesk.seed_align cimport seed_and_extend, get_query_kmers

cdef npc.uint8_t K_LEN = 7
cdef npc.uint8_t K_GAP = 3


cdef void add_range_to_range_list(int range_start, int range_end, list range_list, int slosh):
    """Add a specified range to a list of ranges in place.

    Do this either by 1) merging the range with an existing range
    or 2) appending the new range to the end.
    """
    cdef bint kmer_merged = False
    cdef int i
    for i in range(len(range_list)):
        start, end = range_list[i]
        if range_start <= (end + slosh):
            range_list[i] = (range_list[i][0], range_end)
            kmer_merged = True
            break
    if not kmer_merged:
        range_list.append((range_start, range_end))


cdef npc.uint64_t[:, :] condense_intervals(int slosh, npc.uint64_t[:, :] matched_pos):
    """Condense a 4 column array of potentially overlapping intervals"""
    cdef int n_matched_intervals = 0
    cdef npc.uint64_t[:, :] matching_intervals = np.ndarray((matched_pos.shape[0], 4), dtype=np.uint64)
    cdef int i = 1
    cdef npc.uint64_t current_interval_start_query = matched_pos[0, 0]
    cdef npc.uint64_t current_interval_end_query = matched_pos[0, 1]
    cdef npc.uint64_t current_interval_start_target = matched_pos[0, 2]
    cdef npc.uint64_t current_interval_end_target = matched_pos[0, 3]

    while i < matched_pos.shape[0]:
        query_continues = matched_pos[i, 0] <= (current_interval_end_query + slosh)
        target_continues = matched_pos[i, 2] <= (current_interval_end_target + slosh)
        if query_continues and target_continues:
            current_interval_end_query = matched_pos[i, 1]
            current_interval_end_target = matched_pos[i, 3]
        else:
            matching_intervals[n_matched_intervals, 0] = current_interval_start_query
            matching_intervals[n_matched_intervals, 1] = current_interval_end_query
            matching_intervals[n_matched_intervals, 2] = current_interval_start_target
            matching_intervals[n_matched_intervals, 3] = current_interval_end_target
            n_matched_intervals += 1
            current_interval_start_query = matched_pos[i, 0]
            current_interval_start_target = matched_pos[i, 2]
            current_interval_end_query = matched_pos[i, 1]
            current_interval_end_target = matched_pos[i, 3]
        i += 1

    matching_intervals[n_matched_intervals, 0] = current_interval_start_query
    matching_intervals[n_matched_intervals, 1] = current_interval_end_query
    matching_intervals[n_matched_intervals, 2] = current_interval_start_target
    matching_intervals[n_matched_intervals, 3] = current_interval_end_target
    n_matched_intervals += 1
    return matching_intervals[:n_matched_intervals, :]


cdef class ContigSearcher:
    cdef public ContigDB db
    cdef public double[:, :] centroid_rfts
    cdef public cKDTree tree
    cdef public object logger
    cdef public bint logging
    cdef public float radius

    def __cinit__(self, contig_db, logger=None):
        self.logging = False
        if logger is not None:
            self.logging = True
            self.logger = logger
            self.logger('Loading searcher...')
        self.db = contig_db
        self.db.c_get_centroids()
        self.centroid_rfts = np.ndarray(
            (self.db.cached_centroids.shape[0], self.db.cached_centroids.shape[1])
        )
        if self.logging:
            self.logger(f'Preprocessing centers')
        for i in range(self.db.cached_centroids.shape[0]):
            for j in range(self.db.ramifier.d):
                self.centroid_rfts[i, j] = self.db.cached_centroids[i, j]
                self.centroid_rfts[i, j] *= self.db.box_side_len
                self.centroid_rfts[i, j] += (self.db.box_side_len / 2)
        if self.logging:
            self.logger(f'Building search tree...')
        self.tree = cKDTree(self.centroid_rfts, logger=logger)
        if self.logging:
            self.logger(f'Built search tree.')
        self.radius = (self.db.ramifier.d ** (0.5)) * self.db.box_side_len
        
        
        if self.logging:
            self.logger(f'Searcher loaded. Radius {self.radius}, num. centers {self.centroid_rfts.shape[0]}')

    def py_search(self, str query, double coarse_radius, double kmer_fraction, double identity=0.5):
        return [
            el
            for el in self.search(
                encode_kmer(query), coarse_radius, kmer_fraction, identity
            )
        ]

    cdef list search(self, npc.uint8_t[:] query, double coarse_radius, double kmer_fraction, double identity_thresh):
        if self.logging:
            self.logger(f'Starting query. Coarse radius {coarse_radius}, k-mer fraction {kmer_fraction}')
        cdef npc.uint32_t n_kmers = (query.shape[0] - self.db.ramifier.k + 1) // (self.db.ramifier.k // 2)
        cdef dict centroids_to_query_ranges = self.coarse_search(n_kmers, query, coarse_radius)
        if self.logging:
            self.logger(f'Coarse search complete. {len(centroids_to_query_ranges)} candidates.')
        cdef dict merged_coarse_hits = self.merge_coarse_hits(centroids_to_query_ranges)
        if self.logging:
            self.logger(f'Coarse merge complete.')
        cdef str contig_key
        cdef npc.uint64_t[:, :] matched_intervals
        cdef list out = []
        for contig_key, matched_intervals in merged_coarse_hits.items():
            out += self.fine_search(query, contig_key, matched_intervals, identity_thresh)
        if self.logging:
            self.logger(f'Fine search complete. {len(out)} passed.')
        return out

    cdef dict merge_coarse_hits(self, dict centroids_to_query_ranges):
        cdef dict matched_pos_by_contig = {}
        for centroid_id, query_intervals in centroids_to_query_ranges.items():
            for c_name, c_start, c_end in self.db.get_contigs(centroid_id):
                contig_key = c_name
                if contig_key in matched_pos_by_contig:
                    contig = matched_pos_by_contig[contig_key]
                else:
                    contig = []
                    matched_pos_by_contig[contig_key] = contig
                for q_start, q_end in query_intervals:
                    contig.append((q_start, q_end, c_start, c_end))
        cdef list matched_pos_list
        cdef npc.uint64_t[:, :] matched_pos, buffer_pos
        cdef npc.ndarray order
        for contig_key, matched_pos_list in matched_pos_by_contig.items():
            matched_pos = np.array(matched_pos_list, dtype=np.uint64)[:20,:]
            buffer_pos = np.ndarray((matched_pos.shape[0], matched_pos.shape[1]), dtype=np.uint64)

            order = np.array(matched_pos[:, 2]).argsort(axis=0)
            for i in range(order.shape[0]):
                buffer_pos[i][0] = matched_pos[order[i]][0]
                buffer_pos[i][1] = matched_pos[order[i]][1]
                buffer_pos[i][2] = matched_pos[order[i]][2]
                buffer_pos[i][3] = matched_pos[order[i]][3]
            order = np.array(buffer_pos[:, 0]).argsort(axis=0, kind='stable')
            for i in range(order.shape[0]):
                matched_pos[i][0] = buffer_pos[order[i]][0]
                matched_pos[i][1] = buffer_pos[order[i]][1]
                matched_pos[i][2] = buffer_pos[order[i]][2]
                matched_pos[i][3] = buffer_pos[order[i]][3]

            matched_pos = np.array(matched_pos, dtype=np.uint64)[order]
            matched_pos = condense_intervals(self.db.ramifier.k, matched_pos)
            # print(np.array(matched_pos))
            matched_pos_by_contig[contig_key] = matched_pos
        return matched_pos_by_contig

    cdef list fine_search(self, npc.uint8_t[:] query, str contig_name, npc.uint64_t[:, :] matched_pos, double perc_id_thresh):
        cdef int interval_i
        cdef list out = []
        cdef npc.uint8_t[:] qseq, tseq
        cdef int qstart, qend, tstart, tend
        cdef int slop = self.db.ramifier.k
        cdef StripedSmithWaterman aligner
        cdef double align_score
        for interval_i in range(matched_pos.shape[0]):
            qstart = 0
            if slop < matched_pos[interval_i, 0]:
                qstart = matched_pos[interval_i, 0] - slop
            qend = min(query.shape[0], matched_pos[interval_i, 1] + slop)
            qseq = query[qstart:qend]
            tstart = 0
            if matched_pos[interval_i, 2] > slop:
                tstart = matched_pos[interval_i, 2] - slop
            tend = matched_pos[interval_i, 3] + slop
            if (qend - qstart) < self.db.ramifier.k or (tend - tstart) < self.db.ramifier.k:
                continue
            tseq = self.db.get_seq(contig_name, tstart, tend)
            aligner = StripedSmithWaterman(qseq)
            align_score = aligner.align(tseq)
            #align_score = needle_dist(qseq, tseq, True)
            if align_score >= perc_id_thresh:
                out.append((contig_name, align_score, qstart, qend, tstart, tend, qseq, tseq))
        return out

    cdef double[:, :] _query_kmers(self, int n_kmers, npc.uint8_t[:] query, int kmer_gap):
        cdef double[:] rft
        cdef double[:, :] rfts = np.ndarray(
            (n_kmers, self.db.ramifier.d)
        )
        cdef int i, j, k_start, k_end
        for i in range(n_kmers):
            k_start = i * kmer_gap
            k_end = k_start + self.db.ramifier.k
            rft = self.db.ramifier.c_ramify(query[k_start:k_end])
            for j in range(self.db.ramifier.d):
                rfts[i, j] = rft[j]
        return rfts

    cdef dict coarse_search(self, int n_kmers, npc.uint8_t[:] query, double coarse_radius):
        cdef int kmer_gap = self.db.ramifier.k // 2
        cdef double[:, :] rfts = self._query_kmers(n_kmers, query, kmer_gap)
        cdef cKDTree query_tree = cKDTree(rfts)
        cdef list centroid_hits
        cdef dict centroids_to_query_ranges = {}
        cdef int kmer_i = 0
        cdef int kmer_start, kmer_end, hit
        for centroid_hits in query_tree.query_ball_tree(self.tree, coarse_radius + self.radius):
            kmer_start = kmer_i * kmer_gap
            kmer_end = kmer_start + self.db.ramifier.k
            for hit in centroid_hits:
                if hit in centroids_to_query_ranges:
                    add_range_to_range_list(
                        kmer_start, kmer_end, centroids_to_query_ranges[hit], self.db.ramifier.k
                    )
                else:
                    centroids_to_query_ranges[hit] = [(kmer_start, kmer_end)]
            kmer_i += 1
        return centroids_to_query_ranges

    def search_contigs_from_fasta(self, str filename, double coarse_radius, double kmer_fraction, double identity):
        cdef FILE * cfile = fopen(filename.encode("UTF-8"), "rb")
        if cfile == NULL:
            raise FileNotFoundError(2, "No such file or directory: '%s'" % filename)

        cdef int n_added = 0
        cdef char * line = NULL
        cdef char * header = NULL
        cdef size_t l = 0
        cdef ssize_t read
        cdef size_t n_kmers_in_line, i
        cdef npc.uint8_t[:] seq
        cdef dict out = {}
        while True:
            getline(&header, &l, cfile)
            read = getdelim(&line, &l, b'>', cfile)
            if read == -1: break
            seq = encode_seq_from_buffer(line, l)
            out[str(header)] = self.search(seq, coarse_radius, kmer_fraction, identity)
            n_added += 1
            header = NULL
            line = NULL  # I don't understand why this line is necessary but
                         # without it the program throws a strange error: 
                         # `pointer being realloacted was not allocated`
        fclose(cfile)
        return out

    @classmethod
    def from_filepath(cls, filepath, logger=None):
        return cls(
            ContigDB.load_from_filepath(filepath, logger=logger),
            logger=logger
        )
