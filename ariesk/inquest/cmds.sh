
makeblastdb -in ../1k_fastas.fa -dbtype nucl -out 1k_fastas
/usr/bin/time blastn -db 1k_fastas -out testem -outfmt 6 -query -perc_identity 70 ../sample_query_contigs.fa