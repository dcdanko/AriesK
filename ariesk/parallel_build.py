
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
        cmds.append((temp_filename, cmd))

    n_running = 0
    processes = []
    polled = set()
    while len(cmds) > 0:
        if n_running < threads:
            temp_filename, cmd = cmds.pop()
            process = sp.Popen(cmd, shell=True)
            processes.append((temp_filename, process))
            n_running += 1
        else:
            for i, (temp_filename, process) in enumerate(processes):
                if process.poll() is not None and temp_filename not in polled:
                    polled.add(temp_filename)
                    assert process.returncode == 0
                    logger(i, n_chunks)
                    n_running -= 1
    for _, process in processes:
        process.wait()
    temp_filenames = [el[0] for el in processes]
    cmd = (
        f'{ARIESK_EXC} merge-grid '
        f'{output_filename} '
    ) + ' '.join(temp_filenames)
    run_command((cmd, None))
    for temp_filename in temp_filenames:
        remove(temp_filename)
