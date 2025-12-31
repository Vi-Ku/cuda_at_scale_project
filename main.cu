/*
 * CUDA at Scale: High-Throughput Signal Denoising
 * * Description:
 * This program generates a large volume of synthetic noisy signal data
 * and processes it on the GPU using a 1D Moving Average Filter kernel.
 * It is designed to handle millions of data points to demonstrate "scale".
 */

#include <iostream>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>
#include <string>

// --- CUDA Kernel ---
// Applies a simple moving average filter to smooth data
__global__ void movingAverageKernel(const float* d_input, float* d_output, int numElements, int windowSize) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (idx < numElements) {
        float sum = 0.0f;
        int count = 0;
        
        // Simple 1D convolution loop (centered window)
        for (int i = -windowSize / 2; i <= windowSize / 2; ++i) {
            int neighborIdx = idx + i;
            // Check boundary conditions
            if (neighborIdx >= 0 && neighborIdx < numElements) {
                sum += d_input[neighborIdx];
                count++;
            }
        }
        d_output[idx] = sum / count;
    }
}

// --- Host Helper Functions ---

// Error checking macro
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void printUsage(const char* progName) {
    std::cout << "Usage: " << progName << " [options]\n"
              << "Options:\n"
              << "  -n <int>    Number of signal elements (default: 1,000,000)\n"
              << "  -w <int>    Window size for filter (default: 5)\n"
              << "  -b <int>    Threads per block (default: 256)\n"
              << "  -o <file>   Output CSV file name (default: output.csv)\n";
}

int main(int argc, char** argv) {
    // 1. CLI Argument Parsing
    int numElements = 1000000; // Default: 1 Million data points
    int windowSize = 5;
    int threadsPerBlock = 256;
    std::string outputFile = "output.csv";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-h") || (arg == "--help")) {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "-n" && i + 1 < argc) {
            numElements = std::stoi(argv[++i]);
        } else if (arg == "-w" && i + 1 < argc) {
            windowSize = std::stoi(argv[++i]);
        } else if (arg == "-b" && i + 1 < argc) {
            threadsPerBlock = std::stoi(argv[++i]);
        } else if (arg == "-o" && i + 1 < argc) {
            outputFile = argv[++i];
        }
    }

    std::cout << "--- Configuration ---\n";
    std::cout << "Elements: " << numElements << "\n";
    std::cout << "Window:   " << windowSize << "\n";
    std::cout << "Threads:  " << threadsPerBlock << "\n";
    std::cout << "---------------------\n";

    // 2. Data Generation (Host)
    size_t size = numElements * sizeof(float);
    std::vector<float> h_input(numElements);
    std::vector<float> h_output(numElements);

    // Generate noisy sine wave
    for (int i = 0; i < numElements; ++i) {
        float noise = static_cast<float>(rand()) / RAND_MAX; // 0.0 to 1.0
        h_input[i] = sin(i * 0.01f) + noise; 
    }

    // 3. Memory Allocation (Device)
    float *d_input, *d_output;
    cudaCheckError(cudaMalloc((void**)&d_input, size));
    cudaCheckError(cudaMalloc((void**)&d_output, size));

    // 4. Data Transfer (Host -> Device)
    cudaCheckError(cudaMemcpy(d_input, h_input.data(), size, cudaMemcpyHostToDevice));

    // 5. Kernel Execution
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    std::cout << "Launching Kernel with " << blocksPerGrid << " blocks...\n";
    
    // Start Timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    movingAverageKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, numElements, windowSize);
    
    cudaCheckError(cudaGetLastError());
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // 6. Data Transfer (Device -> Host)
    cudaCheckError(cudaMemcpy(h_output.data(), d_output, size, cudaMemcpyDeviceToHost));

    std::cout << "Processing Complete.\n";
    std::cout << "GPU Time: " << milliseconds << " ms\n";

    // 7. Output Results (Proof of execution)
    // We output only the first 100 points to CSV to save space, 
    // but the processing happened on the full dataset.
    std::ofstream outFile(outputFile);
    outFile << "Index,NoisyInput,SmoothedOutput\n";
    for (int i = 0; i < 100 && i < numElements; ++i) {
        outFile << i << "," << h_input[i] << "," << h_output[i] << "\n";
    }
    outFile.close();
    std::cout << "Sample results written to " << outputFile << "\n";

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}