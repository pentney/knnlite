#!/usr/bin/env python3
"""
Benchmark Results Parser for KNNLite HNSW Implementation

This script parses the output from the FORTRAN benchmark and extracts
key performance metrics in a format suitable for comparison with
other language implementations.
"""

import re
import json
import sys
from typing import Dict, List, Any

def parse_benchmark_output(output_text: str) -> Dict[str, Any]:
    """Parse benchmark output and extract performance metrics."""
    
    results = {
        "implementation": "FORTRAN HNSW",
        "benchmarks": {
            "model_building": [],
            "search_performance": [],
            "accuracy": {},
            "memory_usage": []
        }
    }
    
    lines = output_text.strip().split('\n')
    
    # Parse model building performance
    in_build_section = False
    for line in lines:
        if "BENCHMARK 1: Model Building Performance" in line:
            in_build_section = True
            continue
        elif "BENCHMARK 2:" in line:
            in_build_section = False
            continue
        elif in_build_section and "|" in line and not "Dimension" in line and not "---" in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 4:
                try:
                    dimension = int(parts[0])
                    data_size = int(parts[1])
                    build_time = float(parts[2])
                    build_rate = float(parts[3].replace(',', ''))
                    
                    results["benchmarks"]["model_building"].append({
                        "dimension": dimension,
                        "data_size": data_size,
                        "avg_build_time_seconds": build_time,
                        "build_rate_points_per_second": build_rate
                    })
                except ValueError:
                    continue
    
    # Parse search performance
    in_search_section = False
    for line in lines:
        if "BENCHMARK 2: Search Performance" in line:
            in_search_section = True
            continue
        elif "BENCHMARK 3:" in line:
            in_search_section = False
            continue
        elif in_search_section and "|" in line and not "Dimension" in line and not "---" in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 4:
                try:
                    dimension = int(parts[0])
                    data_size = int(parts[1])
                    search_time = float(parts[2])
                    search_rate = float(parts[3].replace(',', ''))
                    
                    results["benchmarks"]["search_performance"].append({
                        "dimension": dimension,
                        "data_size": data_size,
                        "avg_search_time_seconds": search_time,
                        "search_rate_queries_per_second": search_rate
                    })
                except ValueError:
                    continue
    
    # Parse accuracy
    for line in lines:
        if "Accuracy on synthetic dataset:" in line:
            match = re.search(r'(\d+\.?\d*)%', line)
            if match:
                results["benchmarks"]["accuracy"]["synthetic_dataset_accuracy_percent"] = float(match.group(1))
        
        if "Correct predictions:" in line:
            match = re.search(r'(\d+) out of (\d+)', line)
            if match:
                results["benchmarks"]["accuracy"]["correct_predictions"] = int(match.group(1))
                results["benchmarks"]["accuracy"]["total_predictions"] = int(match.group(2))
    
    # Parse memory usage
    in_memory_section = False
    for line in lines:
        if "BENCHMARK 4: Memory Usage Estimation" in line:
            in_memory_section = True
            continue
        elif in_memory_section and "|" in line and not "Dimension" in line and not "---" in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 4:
                try:
                    dimension = int(parts[0])
                    data_size = int(parts[1])
                    memory_mb = float(parts[2])
                    memory_per_point = float(parts[3])
                    
                    results["benchmarks"]["memory_usage"].append({
                        "dimension": dimension,
                        "data_size": data_size,
                        "estimated_memory_mb": memory_mb,
                        "memory_per_point_bytes": memory_per_point
                    })
                except ValueError:
                    continue
    
    return results

def generate_summary_table(results: Dict[str, Any]) -> str:
    """Generate a summary table of key metrics."""
    
    output = []
    output.append("=" * 80)
    output.append("FASTKNN HNSW BENCHMARK SUMMARY")
    output.append("=" * 80)
    output.append("")
    
    # Model building summary
    output.append("MODEL BUILDING PERFORMANCE:")
    output.append("-" * 40)
    build_data = results["benchmarks"]["model_building"]
    if build_data:
        # Find best and worst performance
        best_build = max(build_data, key=lambda x: x["build_rate_points_per_second"])
        worst_build = min(build_data, key=lambda x: x["build_rate_points_per_second"])
        
        output.append(f"Best build rate: {best_build['build_rate_points_per_second']:,.0f} points/s")
        output.append(f"  (dim={best_build['dimension']}, size={best_build['data_size']})")
        output.append(f"Worst build rate: {worst_build['build_rate_points_per_second']:,.0f} points/s")
        output.append(f"  (dim={worst_build['dimension']}, size={worst_build['data_size']})")
    output.append("")
    
    # Search performance summary
    output.append("SEARCH PERFORMANCE:")
    output.append("-" * 40)
    search_data = results["benchmarks"]["search_performance"]
    if search_data:
        best_search = max(search_data, key=lambda x: x["search_rate_queries_per_second"])
        worst_search = min(search_data, key=lambda x: x["search_rate_queries_per_second"])
        
        output.append(f"Best search rate: {best_search['search_rate_queries_per_second']:,.0f} queries/s")
        output.append(f"  (dim={best_search['dimension']}, size={best_search['data_size']})")
        output.append(f"Worst search rate: {worst_search['search_rate_queries_per_second']:,.0f} queries/s")
        output.append(f"  (dim={worst_search['dimension']}, size={worst_search['data_size']})")
    output.append("")
    
    # Accuracy summary
    output.append("ACCURACY:")
    output.append("-" * 40)
    accuracy = results["benchmarks"]["accuracy"]
    if "synthetic_dataset_accuracy_percent" in accuracy:
        output.append(f"Synthetic dataset accuracy: {accuracy['synthetic_dataset_accuracy_percent']:.1f}%")
    if "correct_predictions" in accuracy:
        output.append(f"Correct predictions: {accuracy['correct_predictions']}/{accuracy['total_predictions']}")
    output.append("")
    
    return "\n".join(output)

def main():
    """Main function to parse benchmark results."""
    
    if len(sys.argv) > 1:
        # Read from file
        with open(sys.argv[1], 'r') as f:
            output_text = f.read()
    else:
        # Read from stdin
        output_text = sys.stdin.read()
    
    # Parse results
    results = parse_benchmark_output(output_text)
    
    # Output JSON
    print(json.dumps(results, indent=2))
    
    # Output summary
    print("\n" + generate_summary_table(results))

if __name__ == "__main__":
    main()
