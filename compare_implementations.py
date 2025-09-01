#!/usr/bin/env python3
"""
Implementation Comparison Tool for KNNLite Benchmarks

This script helps compare benchmark results across different language
implementations of the HNSW algorithm.
"""

import json
import sys
import argparse
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np

def load_results(filename: str) -> Dict[str, Any]:
    """Load benchmark results from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)

def compare_build_performance(results_list: List[Dict[str, Any]]) -> None:
    """Compare model building performance across implementations."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Building Performance Comparison', fontsize=16)
    
    dimensions = [2, 10, 50, 100]
    
    for i, dim in enumerate(dimensions):
        ax = axes[i//2, i%2]
        
        for results in results_list:
            impl_name = results["implementation"]
            build_data = results["benchmarks"]["model_building"]
            
            # Filter data for this dimension
            dim_data = [d for d in build_data if d["dimension"] == dim]
            
            if dim_data:
                sizes = [d["data_size"] for d in dim_data]
                rates = [d["build_rate_points_per_second"] for d in dim_data]
                
                ax.plot(sizes, rates, marker='o', label=impl_name, linewidth=2)
        
        ax.set_xlabel('Data Size')
        ax.set_ylabel('Build Rate (points/s)')
        ax.set_title(f'Dimension {dim}')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('build_performance_comparison.png', dpi=300, bbox_inches='tight')
    print("Build performance comparison saved to build_performance_comparison.png")

def compare_search_performance(results_list: List[Dict[str, Any]]) -> None:
    """Compare search performance across implementations."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Search Performance Comparison', fontsize=16)
    
    dimensions = [2, 10, 50, 100]
    
    for i, dim in enumerate(dimensions):
        ax = axes[i//2, i%2]
        
        for results in results_list:
            impl_name = results["implementation"]
            search_data = results["benchmarks"]["search_performance"]
            
            # Filter data for this dimension
            dim_data = [d for d in search_data if d["dimension"] == dim]
            
            if dim_data:
                sizes = [d["data_size"] for d in dim_data]
                rates = [d["search_rate_queries_per_second"] for d in dim_data]
                
                ax.plot(sizes, rates, marker='o', label=impl_name, linewidth=2)
        
        ax.set_xlabel('Data Size')
        ax.set_ylabel('Search Rate (queries/s)')
        ax.set_title(f'Dimension {dim}')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('search_performance_comparison.png', dpi=300, bbox_inches='tight')
    print("Search performance comparison saved to search_performance_comparison.png")

def generate_comparison_table(results_list: List[Dict[str, Any]]) -> str:
    """Generate a comparison table of key metrics."""
    
    output = []
    output.append("=" * 100)
    output.append("FASTKNN IMPLEMENTATION COMPARISON")
    output.append("=" * 100)
    output.append("")
    
    # Header
    output.append(f"{'Implementation':<20} {'Best Build Rate':<20} {'Best Search Rate':<20} {'Accuracy':<15}")
    output.append("-" * 100)
    
    for results in results_list:
        impl_name = results["implementation"]
        
        # Find best build rate
        build_data = results["benchmarks"]["model_building"]
        best_build = max(build_data, key=lambda x: x["build_rate_points_per_second"])["build_rate_points_per_second"] if build_data else 0
        
        # Find best search rate
        search_data = results["benchmarks"]["search_performance"]
        best_search = max(search_data, key=lambda x: x["search_rate_queries_per_second"])["search_rate_queries_per_second"] if search_data else 0
        
        # Get accuracy
        accuracy = results["benchmarks"]["accuracy"].get("synthetic_dataset_accuracy_percent", 0)
        
        output.append(f"{impl_name:<20} {best_build:<20,.0f} {best_search:<20,.0f} {accuracy:<15.1f}%")
    
    output.append("")
    return "\n".join(output)

def main():
    """Main function for comparison tool."""
    
    parser = argparse.ArgumentParser(description='Compare KNNLite benchmark results')
    parser.add_argument('files', nargs='+', help='JSON result files to compare')
    parser.add_argument('--plots', action='store_true', help='Generate comparison plots')
    parser.add_argument('--table', action='store_true', help='Generate comparison table')
    
    args = parser.parse_args()
    
    # Load all results
    results_list = []
    for filename in args.files:
        try:
            results = load_results(filename)
            results_list.append(results)
            print(f"Loaded results from {filename}")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    if not results_list:
        print("No valid result files loaded")
        return
    
    # Generate comparison table
    if args.table or not args.plots:
        print(generate_comparison_table(results_list))
    
    # Generate plots
    if args.plots:
        try:
            compare_build_performance(results_list)
            compare_search_performance(results_list)
        except ImportError:
            print("matplotlib not available, skipping plots")
        except Exception as e:
            print(f"Error generating plots: {e}")

if __name__ == "__main__":
    main()
