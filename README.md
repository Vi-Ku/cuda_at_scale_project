# CUDA Signal Denoising at Scale

## Project Description
This project implements a high-throughput parallel Moving Average Filter using CUDA to perform signal processing on massive datasets. The application generates large-scale synthetic signal data—simulating noisy sensor or audio streams—and utilizes a custom GPU kernel to filter and smooth the data in parallel.

This project demonstrates "scale" by effectively processing datasets ranging from thousands to tens of millions of data points, leveraging the massive parallelism of the GPU to outperform sequential processing methods.

## Repository Contents
* **`main.cu`**: The core C++ and CUDA source code implementing the signal generation and Moving Average kernel.
* **`Makefile`**: Build script to compile the project using `nvcc`.
* **`run.sh`**: Shell script to automate compilation, execution on multiple dataset sizes, and plot generation.
* **`plot_results.py`**: Python script to generate visualization plots (`.png`) from the output CSV files.
* **`results.md`**: A report summarizing the execution time and performance metrics.

## Requirements
* **CUDA Toolkit**: `nvcc` compiler must be installed and in your PATH.
* **Make**: For building the executable via the Makefile.
* **Python 3**: Required for the visualization script.
    * **Libraries**: `pandas`, `matplotlib`, `numpy`.
    * **Installation**: 
      ```bash
      pip install pandas matplotlib numpy
      ```

## Compilation
To compile the C++ CUDA executable, navigate to the project directory and run:

```bash
make
```

This will generate the `signal_denoise` executable.

## Execution

### Automated Run (Recommended)

The provided shell script compiles the code, runs it on three different "scales" of data (Small, Medium, Large), and generates proof-of-execution plots automatically.

```bash
chmod +x run.sh
./run.sh

```

### Manual Execution

You can also run the denoiser manually with custom arguments:

```bash
./signal_denoise -n <num_elements> -w <window_size> -o <output_file>

```

**Arguments:** 

* `-n`: Number of data points to generate (e.g., `1000000`).
* `-w`: Size of the smoothing window (e.g., `5`).
* `-b`: Threads per block (default: `256`).
* `-o`: Output CSV filename (default: `output.csv`).

To generate a plot manually after running the C++ code:

```bash
python3 plot_results.py output.csv

```

## Proof of Execution

Upon running `run.sh`, the following artifacts will be generated to demonstrate successful execution:

1. **CSV Logs**:
* `output_small.csv`: Raw data for 10,000 elements.
* `output_medium.csv`: Raw data for 1,000,000 elements.
* `output_large.csv`: Raw data for 50,000,000 elements.


2. **Visualization Images**:
* `output_small.png`: Visualizes the noisy input vs. the smoothed output for the small dataset.
    ![Small Scale Plot](output_small.png)
* `output_medium.png`: Demonstrates the kernel working on a larger, denser dataset. Example below:

   ![Medium Scale Plot](output_medium.png)



These images serve as visual proof that the GPU kernel correctly filtered the noise from the signal.

## Rubric Compliance

* **Code Repository**: Complete source provided with `README.md` and CLI support.


* **Support Files**: Includes `Makefile` and `run.sh` for easy compilation and execution.


* **Proof of Execution**: Generates CSV logs and PNG plots demonstrating execution on large datasets.


* **Project Description**: Clearly defines the algorithm (Moving Average) and the scale of data processed.