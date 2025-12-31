#!/bin/bash

# Compile the code
echo "Compiling..."
make

# 1. Run on a "Small" dataset (10,000 elements)
echo "Running Test 1: Small Scale (10k elements)..."
./signal_denoise -n 10000 -w 3 -o output_small.csv

# 2. Run on a "Medium" dataset (1,000,000 elements)
echo "Running Test 2: Medium Scale (1 Million elements)..."
./signal_denoise -n 1000000 -w 5 -o output_medium.csv

# 3. Run on a "Large" dataset (50,000,000 elements)
echo "Running Test 3: Large Scale (50 Million elements)..."
./signal_denoise -n 50000000 -w 10 -o output_large.csv

echo "All tests complete. Check output_*.csv files for results."

echo "Generating Plots..."
python3 plot_results.py output_small.csv
python3 plot_results.py output_medium.csv
# Note: Ensure large csv isn't too huge for matplotlib, 
# or the C++ code only wrote a sample of it.

echo "Done! Check .png files for visual proof."