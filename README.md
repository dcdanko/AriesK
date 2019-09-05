# AriesK

K-mer phylogeny based on the Ramanujan Fourier Transform

## Installation

```
python setup.py build_ext --inplace  # build the cython extensions
python setup.py develop              # install the CLI
python -m pytest                     # run unit tests
```


## Performance

You can run a test evaluation on your machine
```
$ ariesk eval --num-kmers 100000
Made 100000 E. coli k-mers for testing
Build time: 17.914s
{'num_kmers': 100000, 'num_singletons': 12452, 'num_clusters': 25336}
```

This command will build an index from 100,000 32-mers drawn from the E. coli genome. Additional parameters let you change the radius of the index and the k-mer size.