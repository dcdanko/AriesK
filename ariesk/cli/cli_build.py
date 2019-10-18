
import click
from json import dumps

from ariesk.ram import (
    StatisticalRam,
)

from .cli_build_contig import build_contig_cli
from .cli_build_kmer import build_kmer_cli


@click.group('build')
def build_cli():
    pass


build_cli.add_command(build_contig_cli)
build_cli.add_command(build_kmer_cli)


@build_cli.command('rotation')
@click.option('-k', '--kmer-len', default=31)
@click.option('-n', '--num-kmers', default=1000, help='Number of kmers to compare.')
@click.option('-o', '--outfile', default='-', type=click.File('w'))
@click.argument('kmer_table', type=click.Path('r'))
def calculate_pca_rotation(kmer_len, num_kmers, outfile, kmer_table):
    """Calculate a PCA rotation from a set of k-mers."""
    stat_ram = StatisticalRam(kmer_len, num_kmers)
    stat_ram.add_kmers_from_file(kmer_table)
    out = {
        'k': kmer_len,
        'center': stat_ram.get_centers().tolist(),
        'scale': stat_ram.get_scales().tolist(),
        'rotation': stat_ram.get_rotation().tolist(),
    }
    outfile.write(dumps(out))


@build_cli.command('rotation-fasta')
@click.option('-k', '--kmer-len', default=31)
@click.option('-d', '--dropout', default=1000, help='Only keep every nth kmer (millionths)')
@click.option('-n', '--num-kmers', default=1000, help='Number of kmers to compare.')
@click.option('-o', '--outfile', default='-', type=click.File('w'))
@click.argument('fasta_list', type=click.File('r'))
def calculate_pca_rotation_fasta(kmer_len, dropout, num_kmers, outfile, fasta_list):
    """Calculate a PCA rotation from a set of k-mers."""
    stat_ram = StatisticalRam(kmer_len, num_kmers)
    fasta_list = [line.strip() for line in fasta_list]
    with click.progressbar(fasta_list) as fastas:
        for fasta_filename in fastas:
            try:
                stat_ram.fast_add_kmers_from_fasta(fasta_filename, dropout=dropout)
            except IndexError:
                break
    out = {
        'k': kmer_len,
        'center': stat_ram.get_centers().tolist(),
        'scale': stat_ram.get_scales().tolist(),
        'rotation': stat_ram.get_rotation().tolist(),
    }
    outfile.write(dumps(out))
