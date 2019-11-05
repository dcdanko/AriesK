# AriesK

Intrinsic conversion of DNA to vectors based on the Ramanujan Fourier Transform

## Installation

```
python setup.py build_ext --inplace  # build the cython extensions
python setup.py develop              # install the CLI
python -m pytest                     # run unit tests
```


## Building an Implicit Distance Matrix

```
ariesk dists ram -o output.csv <fasta file>
```


## Building and Searching a Contig Database

```
ariesk build rotation-fasta -k 256 -n 100000 -d 1000 -o rotation.json <(ls -1 <fasta file>)
ariesk build contigs from-fasta -o mydb.sqlite rotation.json <(ls -1 <fasta file>)
ariesk search contig-fasta -v -n 3 -r 0.4 mydb.sqlite <fasta file>
```
