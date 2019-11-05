

# Distance Matrix Building

Running on my laptop, nothing else actively running but a few apps open

Runtime
```
61.92   4.18
507.54  5.50
1984.14 108.34
```

```
[     ,Full    ,ariesk]
[70, 95, 5  , 4],
[70, 286, 36 , 34],
[70, 549, 461, 427],
[80, 95, 5  , 4],
[80, 286, 32 , 30],
[80, 549, 295, 269],
[90, 95, 4  , 4],
[90, 286, 30 , 30],
[90, 549, 289, 267],
```

### Full Matrix

#### Small
```
(base) dcdanko@mac196156:~/Dev/ariesk/amrs
$ /usr/bin/time ariesk dists lev -o 20_contigs_dists.lev.csv -g 100 -k 512 20_contigs.fa
95 unique kmers.
60.106s to build distance matrix.
       61.92 real        59.29 user         0.53 sys
(base) dcdanko@mac196156:~/Dev/ariesk/amrs
$ cat 20_contigs_dists.lev.csv | awk -F ',' '{if($4 <= 153) print $0}' | wc -l
       5
```

#### Medium
```
$ /usr/bin/time ariesk dists lev -o 50_contigs_dists.lev.csv -g 100 -k 512 50_contigs.fa
286 unique kmers.
504.64s to build distance matrix.
      507.54 real       502.63 user         1.98 sys
$ /usr/bin/time ariesk dists lev -o 50_contigs_dists.lev.csv -g 100 -k 512 50_contigs.fa
286 unique kmers.
503.01s to build distance matrix.
      505.66 real       501.43 user         1.82 sys
$ cat 50_contigs_dists.lev.csv | awk -F ',' '{if($4 <= 153) print $0}' | wc -l
      36
```

#### Big
```
(base) dcdanko@mac196156:~/Dev/ariesk/amrs
$ /usr/bin/time ariesk dists lev -o 100_contigs_dists.lev.csv -g 100 -k 512 100_contigs.fa
549 unique kmers.
1978.7s to build distance matrix.
     1984.14 real      1959.48 user         5.07 sys
(base) dcdanko@mac196156:~/Dev/ariesk/amrs
$ cat 100_contigs_dists.lev.csv | awk -F ',' '{if($4 <= 153) print $0}' | wc -l
     461
```

### Aries

#### Small
```
(base) dcdanko@mac196156:~/Dev/ariesk/amrs
$ /usr/bin/time ariesk dists ram -r 10 -g 100 -k 512 -o 20_contigs_dists.ram.csv 20_contigs.fa
95 unique kmers.
0.057804s to build distance matrix.
        3.47 real         4.18 user         0.53 sys
(base) dcdanko@mac196156:~/Dev/ariesk/amrs
$ cat 20_contigs_dists.ram.csv | awk -F ',' '{if($4 <= 153) print $0}' | wc -l
       4
```

#### Medium
```
$ ariesk dists ram -r 10 -g 100 -k 512 -o 50_contigs_dists.ram.csv 50_contigs.fa && cat 50_contigs_dists.ram.csv | awk -F ',' '{if($4 <= 153) print $0}' | wc -l
286 unique kmers.
1.8757s to build distance matrix.
      34
$ /usr/bin/time ariesk dists ram -r 10 -g 100 -k 512 -o 50_contigs_dists.ram.csv 50_contigs.fa && cat 50_contigs_dists.ram.csv | awk -F ',' '{if($4 <= 153) print $0}' | wc -l
286 unique kmers.
1.7129s to build distance matrix.
        4.69 real         5.50 user         0.52 sys
      34
```

#### Big
```
(base) dcdanko@mac196156:~/Dev/ariesk/amrs
$ /usr/bin/time ariesk dists ram -r 10 -g 100 -k 512 -o 100_contigs_dists.ram.csv 100_contigs.fa
549 unique kmers.
104.52s to build distance matrix.
      108.34 real       108.10 user         0.84 sys
(base) dcdanko@mac196156:~/Dev/ariesk/amrs
$ cat 100_contigs_dists.ram.csv | awk -F ',' '{if($4 <= 153) print $0}' | wc -l
     427
```

## Database Scaling

```
[1,16,940491],
[1,8,28204],
[1,16, 65536],
[1,8, 256],
[10,8,256],
[10,16,65536],
[100,16,65536],
[100,8,256],
```

## Search

```
cr,time,n_hits,tot_bases
0,0.33763s,2,1280
0.1,0.47673s,11,8064
0.2,0.63366s,26,21730
0.4,1.4213s,45,44805
0.8,2.8408s,36,22144
1.6,14.211s,94,75798
```

```
(base) dcdanko@mac196156:~/Dev/ariesk/amrs
$ ariesk search contig-fasta -n 2 -v -i 50 ariesk_megares.r_0_1.d_8.sqlite 5_contigs.fa -o ariesk.db_r_o_1_d_8.alignments_above_50.5_contigs.megares.csv -r 0.1
Search complete in 0.6539s
Search complete in 0.47673s

(base) dcdanko@mac196156:~/Dev/ariesk/amrs
$ ariesk search contig-fasta -n 2 -v -i 50 ariesk_megares.r_0_1.d_8.sqlite 5_contigs.fa -o ariesk.db_r_0_1_d_8.cr_0_2.alignments_above_50.5_contigs.megares.csv -r 0.2 2> ariesk.db_r_0_1_d_8.cr_0_2.alignments_above_50.5_contigs.megares.log
Search complete in 1.4325s
Search complete in 0.63366s

(base) dcdanko@mac196156:~/Dev/ariesk/amrs
$ ariesk search contig-fasta -n 2 -v -i 50 ariesk_megares.r_0_1.d_8.sqlite 5_contigs.fa -o ariesk.db_r_0_1_d_8.cr_0_8.alignments_above_50.5_contigs.megares.csv -r 0.8 2> ariesk.db_r_0_1_d_8.cr_0_8.alignments_above_50.5_contigs.megares.log
&& ariesk search contig-fasta -n 2 -v -i 50 ariesk_megares.r_0_1.d_8.sqlite 5_contigs.fa -o ariesk.db_r_0_1_d_8.cr_1_6.alignments_above_50.5_contigs.megares.csv -r 1.6 2> ariesk.db_r_0_1_d_8.cr_1_6.alignments_above_50.5_contigs.megares.log

(base) dcdanko@mac196156:~/Dev/ariesk/amrs
$ ariesk search contig-fasta -n 2 -v -i 50 ariesk_megares.r_0_1.d_8.sqlite 5_contigs.fa -o ariesk.db_r_0_1_d_8.cr_0.alignments_above_50.5_contigs.megares.csv -r 0 2> ariesk.db_r_0_1_d_8.cr_0.alignments_above_50.5_contigs.megares.log
```









```