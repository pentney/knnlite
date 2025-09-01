# Makefile for KNNLite HNSW implementation

FC = gfortran
FFLAGS = -O3 -Wall -std=f2008
TARGET = test_knnlite
EXAMPLE_TARGET = example_usage
BENCHMARK_TARGET = benchmark_knnlite
MODULE = knnlite
SOURCE = $(MODULE).f90
TEST_SOURCE = test_$(MODULE).f90
EXAMPLE_SOURCE = example_usage.f90
BENCHMARK_SOURCE = benchmark.f90

all: $(TARGET) $(EXAMPLE_TARGET) $(BENCHMARK_TARGET)

$(TARGET): $(SOURCE) $(TEST_SOURCE)
	$(FC) $(FFLAGS) -o $(TARGET) $(SOURCE) $(TEST_SOURCE)

$(EXAMPLE_TARGET): $(SOURCE) $(EXAMPLE_SOURCE)
	$(FC) $(FFLAGS) -o $(EXAMPLE_TARGET) $(SOURCE) $(EXAMPLE_SOURCE)

$(BENCHMARK_TARGET): $(SOURCE) $(BENCHMARK_SOURCE)
	$(FC) $(FFLAGS) -o $(BENCHMARK_TARGET) $(SOURCE) $(BENCHMARK_SOURCE)

clean:
	rm -f $(TARGET) $(EXAMPLE_TARGET) $(BENCHMARK_TARGET) *.mod *.png

test: $(TARGET)
	./$(TARGET)

example: $(EXAMPLE_TARGET)
	./$(EXAMPLE_TARGET)

run-benchmark: $(BENCHMARK_TARGET)
	./$(BENCHMARK_TARGET)

benchmark-json: $(BENCHMARK_TARGET)
	./$(BENCHMARK_TARGET) | python3 parse_benchmark_results.py > benchmark_results.json

benchmark-summary: $(BENCHMARK_TARGET)
	./$(BENCHMARK_TARGET) | python3 parse_benchmark_results.py

.PHONY: all clean test example run-benchmark benchmark-json benchmark-summary
