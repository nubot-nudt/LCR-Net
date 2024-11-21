#!/bin/bash

# Compile cpp subsampling
cd cpp_subsampling
python3 setup.py build_ext --inplace
cd ..

# Compile cpp neighbors
cd cpp_neighbors
python3 setup.py build_ext --inplace
cd ..

# Compile grouping cuda
cd grouping
cd lib
python3 setup.py build_ext --inplace
cd ..
cd ..



cd nms
python3 setup.py build_ext --inplace
cd ..