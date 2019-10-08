
from os.path import join, dirname
from unittest import TestCase
from ariesk.ram import (
    Ramifier,
    StatisticalRam,
    RotatingRamifier,
)

KMER_TABLE = join(dirname(__file__), 'small_annotated_31mer_table.csv')


class TestRamify(TestCase):

    def test_ramify(self):
        ramifier = Ramifier(32)
        rft = ramifier.ramify('ATCGATCGATCGATCGATCGATCGATCGATCG')
        self.assertTrue(len(rft.shape) == 1)
        self.assertTrue(rft.shape[0] == (4 * 32))

    def test_centers(self):
        stat_ram = StatisticalRam(31, 100)
        stat_ram.add_kmers_from_file(KMER_TABLE)
        centers = stat_ram.get_centers()
        self.assertTrue(centers.shape == (4 * 31,))

    def test_scales(self):
        stat_ram = StatisticalRam(31, 100)
        stat_ram.add_kmers_from_file(KMER_TABLE)
        scales = stat_ram.get_scales()
        self.assertTrue(scales.shape == (4 * 31,))

    def test_rotation(self):
        stat_ram = StatisticalRam(31, 100)
        stat_ram.add_kmers_from_file(KMER_TABLE)
        rotation = stat_ram.get_rotation()
        self.assertTrue(rotation.shape == (4 * 31, 4 * 31))

    def test_rotating_ramifier(self):
        stat_ram = StatisticalRam(31, 100)
        stat_ram.add_kmers_from_file(KMER_TABLE)
        centers = stat_ram.get_centers()
        scales = stat_ram.get_scales()
        rotation = stat_ram.get_rotation()
        rotater = RotatingRamifier(31, 8, rotation, centers, scales)
        rft = rotater.ramify('ATCGATCGATCGATCGATCGATCGATCGATC')
        self.assertTrue(rft.shape == (8,))
