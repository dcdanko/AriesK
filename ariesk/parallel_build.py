
import subprocess as sp
from os import remove, environ
from os.path import basename

ARIESK_EXC = environ.get('ARIESK_EXC', 'ariesk')
HUNDREDK = 100 * 1000


def run_command(args):
    cmd, temp_filename = args
    sp.run(cmd, shell=True, check=True)
    return temp_filename


def coordinate_parallel_build(output_filename, kmer_table, rotation,
    threads, start, num_to_add, radius, dimension,
    chunk_size=HUNDREDK, logger=lambda x, y: None):
    cmds = []
    n_chunks = num_to_add // chunk_size
    for chunk_num in range(n_chunks):
        chunk_start = start + (chunk_num * chunk_size)
        temp_filename = f'temp.ariesk_temp_chunk.{chunk_num}.{basename(kmer_table)}.sqlite'
        cmd = (
            f'{ARIESK_EXC} build-grid '
            f'-r {radius} '
            f'-d {dimension} '
            f'-n {chunk_size} '
            f'-s {chunk_start} '
            f'-o {temp_filename} '
            f'--preload '
            f'{rotation} {kmer_table}'
        )
        process = sp.Popen(cmd, shell=True)
        cmds.append((temp_filename, process))

    temp_filenames = []
    for i, (fname, process) in enumerate(cmds):
        process.wait()
        assert process.returncode == 0
        logger(i, n_chunks)
        temp_filenames.append(fname)
    cmd = (
        f'{ARIESK_EXC} merge-grid '
        f'{output_filename} '
    ) + ' '.join(temp_filenames)
    run_command((cmd, None))
    for temp_filename in temp_filenames:
        remove(temp_filename)
