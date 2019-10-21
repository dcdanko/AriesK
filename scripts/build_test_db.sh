python setup.py build_ext --inplace
rm tests/small_grid_cover.sqlite
ariesk build grid -n 100 -r 0.5 -d 8 -o tests/small_grid_cover.sqlite data/rotation_minikraken.json tests/small_31mer_table.csv
ariesk build prebuild-blooms tests/small_grid_cover.sqlite