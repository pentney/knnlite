#!/usr/bin/env python3
"""
Python benchmark for KNNLite HNSW implementation

This benchmark tests the Python wrapper performance and compares it
with the FORTRAN implementation.
"""

import numpy as np
import time
import sys
import os

# Add current directory to path to import knnlite_python
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from knnlite import KNNClassifier, NewKNNClassifier, Classify

def generate_random_data(dim, num_data):
    """Generate random data with some structure."""
    data = np.random.randn(num_data, dim).astype(np.float32)
    labels = np.array([f"Cluster{(i % 5) + 1}" for i in range(num_data)], dtype='U32')
    
    # Add cluster structure
    for i in range(num_data):
        cluster = (i % 5) + 1
        data[i] += np.array([cluster * 2.0] * dim)
    
    return data, labels

def generate_query_vectors(dim, num_queries):
    """Generate random query vectors."""
    return np.random.rand(num_queries, dim).astype(np.float32) * 10.0

def benchmark_python():
    """Run Python benchmark."""
    # Benchmark parameters
    num_trials = 3
    k = 10
    num_queries = 500
    
    # Test dimensions and sizes
    test_dims = [2, 10, 50, 100]
    test_sizes = [1000, 5000, 10000]
    
    print("========================================")
    print("KNNLite HNSW Benchmark Suite (Python)")
    print("========================================")
    print("Testing HNSW model building and search performance")
    print(f"Number of trials per test: {num_trials}")
    print(f"K for classification: {k}")
    print(f"Number of queries per test: {num_queries}")
    print()
    
    # Benchmark 1: Model Building Performance
    print("BENCHMARK 1: Model Building Performance")
    print("=======================================")
    print("Dimension | Data Size | Avg Build Time (s) | Build Rate (points/s)")
    print("----------|-----------|-------------------|---------------------")
    
    for dim in test_dims:
        for size in test_sizes:
            # Generate random data
            data, labels = generate_random_data(dim, size)
            
            total_build_time = 0.0
            
            for trial in range(num_trials):
                start = time.time()
                classifier = NewKNNClassifier(dim, data, labels)
                end = time.time()
                
                build_time = end - start
                total_build_time += build_time
            
            avg_build_time = total_build_time / num_trials
            build_rate = size / avg_build_time
            print(f"{dim:9d} | {size:10d} | {avg_build_time:18.6f} | {build_rate:20.0f}")
        
        print()
    
    # Benchmark 2: Search Performance
    print("BENCHMARK 2: Search Performance")
    print("==============================")
    print("Dimension | Data Size | Avg Search Time (s) | Search Rate (queries/s)")
    print("----------|-----------|-------------------|----------------------")
    
    for dim in test_dims:
        for size in test_sizes:
            # Generate random data
            data, labels = generate_random_data(dim, size)
            
            # Build classifier once
            classifier = NewKNNClassifier(dim, data, labels)
            
            # Generate query vectors
            query_vectors = generate_query_vectors(dim, num_queries)
            
            total_search_time = 0.0
            
            for trial in range(num_trials):
                start = time.time()
                predictions = classifier.predict(query_vectors, k)
                end = time.time()
                
                search_time = end - start
                total_search_time += search_time
            
            avg_search_time = total_search_time / num_trials
            search_rate = num_queries / avg_search_time
            print(f"{dim:9d} | {size:10d} | {avg_search_time:18.6f} | {search_rate:22.0f}")
        
        print()
    
    # Benchmark 3: Accuracy Test
    print("BENCHMARK 3: Classification Accuracy")
    print("===================================")
    print("Testing accuracy on synthetic dataset with known clusters")
    print()
    
    # Generate structured data with clear clusters
    dim = 2
    num_data = 1000
    num_queries = 100
    k = 5
    
    data = np.random.randn(num_data, dim).astype(np.float32)
    labels = np.array([f"Class{(i % 3) + 1}" for i in range(num_data)], dtype='U32')
    
    # Add cluster structure
    for i in range(num_data):
        cluster = (i % 3) + 1
        data[i] += np.array([cluster * 3.0, cluster * 3.0])
    
    # Build classifier
    classifier = NewKNNClassifier(dim, data, labels)
    
    # Generate test queries
    num_correct = 0
    for i in range(num_queries):
        cluster = (i % 3) + 1
        
        # Generate query near cluster center
        query = np.array([cluster * 3.0, cluster * 3.0], dtype=np.float32)
        query += np.random.randn(2).astype(np.float32) * 0.5
        
        true_label = f"Class{cluster}"
        
        # Classify
        predictions = classifier.predict(query.reshape(1, -1), k)
        
        # Check accuracy
        if predictions[0] == true_label:
            num_correct += 1
    
    accuracy = num_correct / num_queries * 100.0
    print(f"Accuracy on synthetic dataset: {accuracy:.1f}%")
    print(f"Correct predictions: {num_correct} out of {num_queries}")
    print()
    
    # Benchmark 4: Memory Usage (approximate)
    print("BENCHMARK 4: Memory Usage Estimation")
    print("===================================")
    print("Dimension | Data Size | Est. Memory (MB) | Memory per Point (bytes)")
    print("----------|-----------|-----------------|------------------------")
    
    for dim in test_dims:
        for size in test_sizes:
            # Estimate memory usage (Python overhead)
            estimated_memory = size * (dim * 4.0 + 32.0 + 4.0) / (1024.0 * 1024.0)
            memory_per_point = estimated_memory * 1024.0 * 1024.0 / size
            print(f"{dim:9d} | {size:10d} | {estimated_memory:15.2f} | {memory_per_point:25.1f}")
        
        print()
    
    print("========================================")
    print("Benchmark completed successfully!")
    print("========================================")

if __name__ == "__main__":
    benchmark_python()
