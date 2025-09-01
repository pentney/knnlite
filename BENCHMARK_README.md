# KNNLite HNSW Benchmark Suite

This benchmark suite tests the performance of the FORTRAN HNSW implementation for k-nearest neighbor classification. The benchmark is designed to be language-agnostic and produce results that can be easily compared with implementations in other languages.

## Benchmark Structure

The benchmark consists of four main tests:

### 1. Model Building Performance
Tests the time required to construct the HNSW graph structure from training data.

**Parameters:**
- Dimensions: 2, 10, 50, 100
- Data sizes: 1,000, 5,000, 10,000 points
- Number of trials: 3 (averaged)
- Metric: Build rate (points/second)

### 2. Search Performance
Tests the time required to perform k-nearest neighbor searches on the constructed HNSW graph.

**Parameters:**
- Dimensions: 2, 10, 50, 100
- Data sizes: 1,000, 5,000, 10,000 points
- Number of queries: 500 per test
- K value: 10
- Number of trials: 3 (averaged)
- Metric: Search rate (queries/second)

### 3. Classification Accuracy
Tests the accuracy of the k-NN classifier on a synthetic dataset with known clusters.

**Parameters:**
- Dimension: 2
- Data size: 1,000 points
- Number of queries: 100
- K value: 5
- Clusters: 3 well-separated clusters

### 4. Memory Usage Estimation
Provides estimates of memory consumption for different dataset sizes and dimensions.

## Running the Benchmark

### Basic Usage
```bash
# Compile and run benchmark
make run-benchmark

# Generate JSON results
make benchmark-json

# Generate summary with JSON
make benchmark-summary
```

### Output Formats

1. **Human-readable output**: Direct console output with formatted tables
2. **JSON output**: Machine-readable results for comparison tools
3. **Summary**: Key performance metrics extracted from results

## Sample Results (FORTRAN Implementation)

### Model Building Performance
- **Best**: 14,079 points/s (2D, 1,000 points)
- **Worst**: 1,562 points/s (100D, 10,000 points)

### Search Performance
- **Best**: 177,364 queries/s (2D, 1,000 points)
- **Worst**: 1,354 queries/s (100D, 10,000 points)

### Memory Usage
- **2D data**: ~2.2 KB per point
- **100D data**: ~2.5 KB per point

## Language-Agnostic Design

The benchmark is designed to be easily ported to other languages:

1. **Standard metrics**: Uses universally comparable metrics (time, rate, accuracy)
2. **JSON output**: Machine-readable format for automated comparison
3. **Modular structure**: Each benchmark can be run independently
4. **Clear parameters**: Well-documented test parameters and conditions

## Comparison Tools

### JSON Parser (`parse_benchmark_results.py`)
Extracts structured data from benchmark output for analysis and comparison.

### Implementation Comparator (`compare_implementations.py`)
Compares results across multiple implementations and generates:
- Performance comparison tables
- Visualization plots (if matplotlib available)
- Summary statistics

### Usage Example
```bash
# Compare multiple implementations
python3 compare_implementations.py fortran_results.json python_results.json cpp_results.json --table --plots
```

## Benchmark Parameters

### HNSW Configuration
- **MAX_LAYERS**: 16 (maximum hierarchical layers)
- **M**: 16 (maximum connections per node in upper layers)
- **M_MAX**: 32 (maximum connections per node in layer 0)
- **EF_CONSTRUCTION**: 200 (candidate list size during construction)
- **EF_SEARCH**: 50 (candidate list size during search)

### Test Data Generation
- **Clusters**: 5 synthetic clusters with Gaussian noise
- **Query distribution**: Uniform random distribution
- **Random seed**: Fixed for reproducibility

## Performance Characteristics

### Scaling Behavior
- **Build time**: O(n log n) complexity
- **Search time**: O(log n) complexity
- **Memory usage**: O(n) with constant factor depending on dimension

### Dimension Impact
- Higher dimensions increase both build and search time
- Memory usage scales linearly with dimension
- Search performance degrades more rapidly than build performance

## Reproducibility

The benchmark uses:
- Fixed random seeds for consistent results
- Multiple trials with averaging
- Standardized test parameters
- Clear documentation of all settings

## Extending the Benchmark

To add new test cases or modify parameters:

1. Edit `benchmark.f90` to add new test dimensions or sizes
2. Update the JSON parser if new metrics are added
3. Modify comparison tools for new analysis types
4. Update documentation with new parameters

## Files

- `benchmark.f90`: Main benchmark implementation
- `parse_benchmark_results.py`: Results parser and JSON generator
- `compare_implementations.py`: Multi-implementation comparison tool
- `benchmark_results.json`: Sample results from FORTRAN implementation
- `Makefile`: Build and run configuration
- `BENCHMARK_README.md`: This documentation

## Future Enhancements

Potential improvements to the benchmark suite:

1. **Additional metrics**: Precision, recall, F1-score for classification
2. **More test cases**: Different data distributions, edge cases
3. **Visualization**: Built-in plotting capabilities
4. **Automated testing**: Continuous integration support
5. **Performance profiling**: Memory and CPU usage tracking
6. **Scalability tests**: Larger dataset sizes for performance analysis
