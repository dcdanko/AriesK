
from math import ceil
from .tables import (
    coarse_search_radius,
    subk_filter,
)


class ParameterPicker:

    def __init__(self, ram_dim, k_len, sub_k_len):
        self.ram_dim = ram_dim
        self.k_len = k_len
        self.sub_k_len = sub_k_len

    def coarse_radius(self, max_diff_rate):
        sub_table = coarse_search_radius[(self.ram_dim, self.k_len)]
        max_diffs = int(ceil(max_diff_rate * self.k_len))
        try:
            return sub_table[max_diffs]
        except KeyError:
            return sub_table[-1]

    def min_filter_overlap(self, max_diff_rate):
        sub_table = subk_filter[(self.sub_k_len, self.k_len)]
        max_diffs = int(ceil(max_diff_rate * self.k_len))
        try:
            return sub_table[max_diffs]
        except KeyError:
            return sub_table[-1]
