# Compiler
NVCC = nvcc

# Flags
NVCC_FLAGS = -std=c++11 -O3

# Target executable name
TARGET = signal_denoise

# Build rules
all: $(TARGET)

$(TARGET): main.cu
	$(NVCC) $(NVCC_FLAGS) main.cu -o $(TARGET)

clean:
	rm -f $(TARGET) *.o output.csv *.csv