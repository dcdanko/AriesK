#include <cmath>
#include <cstdlib>
#include <cstring>

#include <vector>
#include <string>
#include <sstream>
#include <new>
#include <typeinfo>
#include <stdexcept>
#include <ios>

#include "ckdtree_decl.h"
#include "rectangle.h"
#include "ckdtree_decl.h"
#include "ordered_pair.h"
#include "partial_sort.h"

#define tree_buffer_root(buf) (&(buf)[0][0])

static ckdtree_intp_t
build(ckdtree *self, ckdtree_intp_t start_idx, intptr_t end_idx,
      double *maxes, double *mins,
      const int _median, const int _compact)
{

    const ckdtree_intp_t m = self->m;
    const double *data = self->raw_data;
    ckdtree_intp_t *indices = (intptr_t *)(self->raw_indices);

    ckdtreenode new_node, *n, *root;
    ckdtree_intp_t node_index, _less, _greater;
    ckdtree_intp_t i, j, p, q, d;
    double size, split, minval, maxval;

    /* put a new node into the node stack */
    self->tree_buffer->push_back(new_node);
    node_index = self->tree_buffer->size() - 1;
    root = tree_buffer_root(self->tree_buffer);
    n = root + node_index;
    memset(n, 0, sizeof(n[0]));

    n->start_idx = start_idx;
    n->end_idx = end_idx;
    n->children = end_idx - start_idx;

    if (end_idx-start_idx <= self->leafsize) {
        /* below brute force limit, return leafnode */
        n->split_dim = -1;
        return node_index;
    }
    else {

        if (CKDTREE_LIKELY(_compact)) {
            /* Recompute hyperrectangle bounds. This should lead to a more
             * compact kd-tree but comes at the expense of larger construction
             * time. However, construction time is usually dwarfed by the
             * query time by orders of magnitude.
             */
            const double *tmp_data_point;
            tmp_data_point = data + indices[start_idx] * m;
            for(i=0; i<m; ++i) {
                maxes[i] = tmp_data_point[i];
                mins[i] = tmp_data_point[i];
            }
            for (j = start_idx + 1; j < end_idx; ++j) {
                tmp_data_point = data + indices[j] * m;
                for(i=0; i<m; ++i) {
                    double tmp = tmp_data_point[i];
                    maxes[i] = maxes[i] > tmp ? maxes[i] : tmp;
                    mins[i] = mins[i] < tmp ? mins[i] : tmp;
                }
            }
        }

        /* split on the dimension with largest spread */
        d = 0;
        size = 0;
        for (i=0; i<m; ++i) {
            if (maxes[i] - mins[i] > size) {
                d = i;
                size = maxes[i] - mins[i];
            }
        }
        maxval = maxes[d];
        minval = mins[d];
        if (maxval == minval) {
            /* all points are identical; warn user?
             * return leafnode
             */
            n->split_dim = -1;
            return node_index;
        }

        /* construct new inner node */

        if (CKDTREE_LIKELY(_median)) {
            /* split on median to create a balanced tree
             * adopted from scikit-learn
             */
            i = (end_idx - start_idx) / 2;
            partition_node_indices(data, indices + start_idx, d, i, m,
                end_idx - start_idx);
            p = start_idx + i;
            split = data[indices[p]*m+d];
        }
        else {
            /* split with the sliding midpoint rule */
            split = (maxval + minval) / 2;
        }

        p = start_idx;
        q = end_idx - 1;
        while (p <= q) {
            if (data[indices[p] * m + d] < split)
                ++p;
            else if (data[indices[q] * m + d] >= split)
                --q;
            else {
                ckdtree_intp_t t = indices[p];
                indices[p] = indices[q];
                indices[q] = t;
                ++p;
                --q;
            }
        }
        /* slide midpoint if necessary */
        if (p == start_idx) {
            /* no points less than split */
            j = start_idx;
            split = data[indices[j] * m + d];
            for (i = start_idx+1; i < end_idx; ++i) {
                if (data[indices[i] * m + d] < split) {
                    j = i;
                    split = data[indices[j] * m + d];
                }
            }
            ckdtree_intp_t t = indices[start_idx];
            indices[start_idx] = indices[j];
            indices[j] = t;
            p = start_idx + 1;
            q = start_idx;
        }
        else if (p == end_idx) {
            /* no points greater than split */
            j = end_idx - 1;
            split = data[indices[j] * m + d];
            for (i = start_idx; i < end_idx-1; ++i) {
                if (data[indices[i] * m + d] > split) {
                    j = i;
                    split = data[indices[j] * m + d];
                }
            }
            ckdtree_intp_t t = indices[end_idx-1];
            indices[end_idx-1] = indices[j];
            indices[j] = t;
            p = end_idx - 1;
            q = end_idx - 2;
        }

        if (CKDTREE_LIKELY(_compact)) {
            _less = build(self, start_idx, p, maxes, mins, _median, _compact);
            _greater = build(self, p, end_idx, maxes, mins, _median, _compact);
        }
        else
        {
            std::vector<double> tmp(m);
            double *mids = &tmp[0];

            for (i=0; i<m; ++i) mids[i] = maxes[i];
            mids[d] = split;
            _less = build(self, start_idx, p, mids, mins, _median, _compact);

            for (i=0; i<m; ++i) mids[i] = mins[i];
            mids[d] = split;
            _greater = build(self, p, end_idx, maxes, mids, _median, _compact);
        }

        /* recompute n because std::vector can
         * reallocate its internal buffer
         */
        root = tree_buffer_root(self->tree_buffer);
        n = root + node_index;
        /* fill in entries */
        n->_less = _less;
        n->_greater = _greater;
        n->less = root + _less;
        n->greater = root + _greater;
        n->split_dim = d;
        n->split = split;

        return node_index;
    }
}



int build_ckdtree(ckdtree *self, ckdtree_intp_t start_idx, intptr_t end_idx,
              double *maxes, double *mins, int _median, int _compact)

{
    build(self, start_idx, end_idx, maxes, mins, _median, _compact);
    return 0;
}

static double
add_weights(ckdtree *self,
           double *node_weights,
           ckdtree_intp_t node_index,
           double *weights)
{

    ckdtree_intp_t *indices = (intptr_t *)(self->raw_indices);

    ckdtreenode *n, *root;

    root = tree_buffer_root(self->tree_buffer);

    n = root + node_index;

    double sum = 0;

    if (n->split_dim != -1) {
        /* internal nodes; recursively calculate the total weight */
        double left, right;
        left = add_weights(self, node_weights, n->_less, weights);
        right = add_weights(self, node_weights, n->_greater, weights);
        sum = left + right;
    } else {
        ckdtree_intp_t i;

        /* Leaf nodes */
        for (i = n->start_idx; i < n->end_idx; ++i) {
            sum += weights[indices[i]];
        }
    }

    node_weights[node_index] = sum;
    return sum;
}

int
build_weights (ckdtree *self, double *node_weights, double *weights)
{

    add_weights(self, node_weights, 0, weights);
    return 0;
}


static void
traverse_no_checking(const ckdtree *self, const ckdtree *other,
                     std::vector<ckdtree_intp_t> **results,
                     const ckdtreenode *node1, const ckdtreenode *node2)
{
    const ckdtreenode *lnode1;
    const ckdtreenode *lnode2;
    const ckdtree_intp_t *sindices = self->raw_indices;
    const ckdtree_intp_t *oindices = other->raw_indices;
    std::vector<ckdtree_intp_t> *results_i;
    ckdtree_intp_t i, j;

    if (node1->split_dim == -1) {   /* leaf node */
        lnode1 = node1;

        if (node2->split_dim == -1) {  /* leaf node */
            lnode2 = node2;

            const ckdtree_intp_t start1 = lnode1->start_idx;
            const ckdtree_intp_t start2 = lnode2->start_idx;
            const ckdtree_intp_t end1 = lnode1->end_idx;
            const ckdtree_intp_t end2 = lnode2->end_idx;

            for (i = start1; i < end1; ++i) {
                results_i = results[sindices[i]];
                for (j = start2; j < end2; ++j)
                    results_i->push_back(oindices[j]);
            }
        }
        else {
            traverse_no_checking(self, other, results, node1, node2->less);
            traverse_no_checking(self, other, results, node1, node2->greater);
        }
    }
    else {
        traverse_no_checking(self, other, results, node1->less, node2);
        traverse_no_checking(self, other, results, node1->greater, node2);
    }
}


template <typename MinMaxDist> static void
traverse_checking(const ckdtree *self, const ckdtree *other,
                  std::vector<ckdtree_intp_t> **results,
                  const ckdtreenode *node1, const ckdtreenode *node2,
                  RectRectDistanceTracker<MinMaxDist> *tracker)
{
    const ckdtreenode *lnode1;
    const ckdtreenode *lnode2;
    std::vector<ckdtree_intp_t> *results_i;
    double d;
    ckdtree_intp_t i, j;

    if (tracker->min_distance > tracker->upper_bound * tracker->epsfac)
        return;
    else if (tracker->max_distance < tracker->upper_bound / tracker->epsfac)
        traverse_no_checking(self, other, results, node1, node2);
    else if (node1->split_dim == -1) { /* 1 is leaf node */
        lnode1 = node1;

        if (node2->split_dim == -1) {  /* 1 & 2 are leaves */

            /* brute-force */
            lnode2 = node2;
            const double p = tracker->p;
            const double tub = tracker->upper_bound;
            const double tmd = tracker->max_distance;
            const double *sdata = self->raw_data;
            const ckdtree_intp_t *sindices = self->raw_indices;
            const double *odata = other->raw_data;
            const ckdtree_intp_t *oindices = other->raw_indices;
            const ckdtree_intp_t m = self->m;
            const ckdtree_intp_t start1 = lnode1->start_idx;
            const ckdtree_intp_t start2 = lnode2->start_idx;
            const ckdtree_intp_t end1 = lnode1->end_idx;
            const ckdtree_intp_t end2 = lnode2->end_idx;

            CKDTREE_PREFETCH(sdata + sindices[start1] * m, 0, m);

            if (start1 < end1 - 1)
                CKDTREE_PREFETCH(sdata + sindices[start1+1] * m, 0, m);

            for (i = start1; i < end1; ++i) {

                if (i < end1 - 2)
                    CKDTREE_PREFETCH(sdata + sindices[i+2] * m, 0, m);

                CKDTREE_PREFETCH(odata + oindices[start2] * m, 0, m);

                if (start2 < end2 - 1)
                    CKDTREE_PREFETCH(odata + oindices[start2+1] * m, 0, m);

                results_i = results[sindices[i]];

                for (j = start2; j < end2; ++j) {

                    if (j < end2 - 2)
                        CKDTREE_PREFETCH(odata + oindices[j+2] * m, 0, m);

                    d = MinMaxDist::point_point_p(
                            self,
                            sdata + sindices[i] * m,
                            odata + oindices[j] * m,
                            p, m, tmd);

                    if (d <= tub)
                        results_i->push_back(other->raw_indices[j]);
                }
            }

        }
        else { /* 1 is a leaf node, 2 is inner node */

            tracker->push_less_of(2, node2);
            traverse_checking(
                self, other, results, node1, node2->less, tracker);
            tracker->pop();

            tracker->push_greater_of(2, node2);
            traverse_checking(
                self, other, results, node1, node2->greater, tracker);
            tracker->pop();
        }
    }
    else {  /* 1 is an inner node */
        if (node2->split_dim == -1) { /* 1 is an inner node, 2 is a leaf node */
            tracker->push_less_of(1, node1);
            traverse_checking(
                self, other, results, node1->less, node2, tracker);
            tracker->pop();

            tracker->push_greater_of(1, node1);
            traverse_checking(
                self, other, results, node1->greater, node2, tracker);
            tracker->pop();
        }
        else { /* 1 & 2 are inner nodes */

            tracker->push_less_of(1, node1);
            tracker->push_less_of(2, node2);
            traverse_checking(
                self, other, results, node1->less, node2->less, tracker);
            tracker->pop();

            tracker->push_greater_of(2, node2);
            traverse_checking(
                self, other, results, node1->less, node2->greater, tracker);
            tracker->pop();
            tracker->pop();


            tracker->push_greater_of(1, node1);
            tracker->push_less_of(2, node2);
            traverse_checking(
                self, other, results, node1->greater, node2->less, tracker);
            tracker->pop();

            tracker->push_greater_of(2, node2);
            traverse_checking(
                self, other, results, node1->greater, node2->greater, tracker);
            tracker->pop();
            tracker->pop();
        }
    }
}

int
query_ball_tree(const ckdtree *self, const ckdtree *other,
                const double r, const double p, const double eps,
                std::vector<ckdtree_intp_t> **results)
{

#define HANDLE(cond, kls) \
    if(cond) { \
        RectRectDistanceTracker<kls> tracker(self, r1, r2, p, eps, r); \
        traverse_checking(self, other, results, self->ctree, other->ctree, \
            &tracker); \
    } else

    Rectangle r1(self->m, self->raw_mins, self->raw_maxes);
    Rectangle r2(other->m, other->raw_mins, other->raw_maxes);

    if(CKDTREE_LIKELY(self->raw_boxsize_data == NULL)) {
        HANDLE(CKDTREE_LIKELY(p == 2), MinkowskiDistP2)
        HANDLE(p == 1, MinkowskiDistP1)
        HANDLE(ckdtree_isinf(p), MinkowskiDistPinf)
        HANDLE(1, MinkowskiDistPp)
        {}
    } else {
        HANDLE(CKDTREE_LIKELY(p == 2), BoxMinkowskiDistP2)
        HANDLE(p == 1, BoxMinkowskiDistP1)
        HANDLE(ckdtree_isinf(p), BoxMinkowskiDistPinf)
        HANDLE(1, BoxMinkowskiDistPp)
        {}
    }
    return 0;
}