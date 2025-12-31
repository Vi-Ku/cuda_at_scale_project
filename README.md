# CUDA Signal Denoising at Scale

## Project Description
This project implements a parallel Moving Average Filter using CUDA to process large-scale signal data. It generates synthetic noisy signals (sine waves with random noise) and cleans them using a GPU kernel. This demonstrates high-throughput data processing capabilities suitable for audio or sensor data streams.

## Requirements
* CUDA Toolkit (nvcc)
* Make

## Compilation
To compile the project, run:
    make

## Execution
You can run the program using the provided shell script to demonstrate scaling:
    ./run.sh

Or run it manually with custom CLI arguments:
    ./signal_denoise -n <num_elements> -w <window_size>

## Arguments
* `-n`: Number of data points to generate (e.g., 1000000)
* `-w`: Size of the smoothing window (e.g., 5)
* `-b`: Threads per block (default 256)
* `-o`: Output CSV file name

## Deliverables
* `main.cu`: Source code.
* `Makefile`: Build system.
* `output_*.csv`: Proof of execution logs showing input vs smoothed output.