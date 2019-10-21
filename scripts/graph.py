import click
import networkx as nx
import matplotlib.pyplot as plt


def encode(base):
    code = 0
    if base == 'A':
        code = 1
    elif base == 'C':
        code = 2
    elif base == 'G':
        code = 3
    elif base == 'T':
        code = 4
    return code


@click.command()
@click.argument('seq_file', type=click.File('r'))
def graph(seq_file):
    seqs = [line.strip() for line in seq_file]
    k = max([len(seq) for seq in seqs])
    grid = [([' '] * k), ([' '] * k), ([' '] * k), ([' '] * k), ([' '] * k)]
    for seq in seqs:
        for i, base in enumerate(seq):
            code = encode(base)
            grid[code][i] = base
    for line in grid:
        print(''.join(line))


@click.command()
@click.option('-o', '--outfile', default='-', type=click.File('w'))
@click.argument('seq_file', type=click.File('r'))
def graph2(outfile, seq_file):
    seqs = [line.strip() for line in seq_file]
    dg = nx.MultiDiGraph()
    start_node = '^'
    n = 1
    for seq in seqs:
        node = start_node
        for i in range(0, len(seq), n):
            codon = (i, seq[i:i + n])
            dg.add_edge(node, codon)
            node = codon
    nx.draw(dg)
    plt.draw()
    plt.show()


if __name__ == '__main__':
    graph2()
