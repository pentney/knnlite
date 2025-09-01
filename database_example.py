#!/usr/bin/env python3
"""
KNNLite Database Integration Example

This script demonstrates how to use the KNNLite database integration
to build HNSW indices from PostgreSQL tables with pgvector support.
"""

import numpy as np
import time
from knnlite_database import DatabaseHNSWBuilder, create_sample_table, benchmark_database_hnsw

def main():
    """Main example function."""
    
    # Database connection parameters
    # Update these to match your PostgreSQL setup
    db_params = {
        'host': 'localhost',
        'port': 5432,
        'database': 'knnlite_test',
        'user': 'postgres',
        'password': 'password'
    }
    
    print("KNNLite Database Integration Example")
    print("=" * 50)
    
    try:
        # Step 1: Create sample data
        print("\n1. Creating sample table with vector data...")
        create_sample_table(
            db_params, 
            table_name='sample_vectors',
            num_samples=10000,
            dimension=128
        )
        
        # Step 2: Build HNSW index from database
        print("\n2. Building HNSW index from database...")
        with DatabaseHNSWBuilder(db_params) as builder:
            # Build index with all data
            classifier = builder.build_hnsw_index(
                'sample_vectors', 
                'vector', 
                'label'
            )
            
            # Get index information
            info = builder.get_index_info()
            print(f"   Index dimension: {info['dimension']}")
            print(f"   Number of nodes: {info['num_nodes']}")
            print(f"   Max layer: {info['max_layer']}")
            
            # Step 3: Test queries
            print("\n3. Testing queries...")
            
            # Generate test queries
            test_vectors = np.random.randn(10, 128).astype(np.float32)
            
            start_time = time.time()
            predictions = builder.query_hnsw(test_vectors, k=5)
            query_time = time.time() - start_time
            
            print(f"   Query time for 10 vectors: {query_time:.4f} seconds")
            print(f"   Query rate: {10/query_time:.0f} queries/second")
            
            # Show some predictions
            print("\n   Sample predictions:")
            for i, (vector, pred) in enumerate(zip(test_vectors[:5], predictions[:5])):
                print(f"   Query {i+1}: {pred}")
        
        # Step 4: Benchmark different data sizes
        print("\n4. Running benchmark...")
        benchmark_database_hnsw(
            db_params,
            table_name='sample_vectors',
            test_sizes=[1000, 5000, 10000],
            test_dimensions=[64, 128]
        )
        
        # Step 5: Demonstrate filtering
        print("\n5. Building index with filtered data...")
        with DatabaseHNSWBuilder(db_params) as builder:
            # Build index only for specific clusters
            classifier = builder.build_hnsw_index(
                'sample_vectors',
                'vector',
                'label',
                where_clause="label IN ('cluster_1', 'cluster_2')"
            )
            
            print(f"   Filtered index nodes: {classifier.num_nodes}")
            
            # Test with filtered data
            test_vectors = np.random.randn(5, 128).astype(np.float32)
            predictions = builder.query_hnsw(test_vectors, k=3)
            
            print("   Predictions on filtered data:")
            for i, pred in enumerate(predictions):
                print(f"   Query {i+1}: {pred}")
        
        print("\n" + "=" * 50)
        print("Example completed successfully!")
        print("\nTo use with your own data:")
        print("1. Create a table with pgvector column")
        print("2. Insert your vector data")
        print("3. Use DatabaseHNSWBuilder to build index")
        print("4. Query the index for nearest neighbors")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure PostgreSQL is running")
        print("2. Install pgvector extension: CREATE EXTENSION vector;")
        print("3. Update database connection parameters")
        print("4. Install required packages: pip install -r requirements_database.txt")

def create_real_world_example():
    """Example with more realistic data structure."""
    
    db_params = {
        'host': 'localhost',
        'port': 5432,
        'database': 'knnlite_test',
        'user': 'postgres',
        'password': 'password'
    }
    
    print("\nReal-world Example: Document Embeddings")
    print("=" * 50)
    
    try:
        with DatabaseHNSWBuilder(db_params) as builder:
            # Example: Build index from document embeddings
            # Assuming you have a table like:
            # CREATE TABLE documents (
            #     id SERIAL PRIMARY KEY,
            #     title VARCHAR(255),
            #     content TEXT,
            #     embedding vector(384),  -- Sentence transformer embedding
            #     category VARCHAR(100)
            # );
            
            print("Building HNSW index from document embeddings...")
            
            # This would work with your actual table
            # classifier = builder.build_hnsw_index(
            #     'documents',
            #     'embedding',
            #     'category',
            #     where_clause="category IS NOT NULL"
            # )
            
            # For demo, use sample data
            classifier = builder.build_hnsw_index(
                'sample_vectors',
                'vector',
                'label'
            )
            
            # Simulate document similarity search
            print("Performing document similarity search...")
            
            # Query vector (e.g., from a search query)
            query_vector = np.random.randn(1, 128).astype(np.float32)
            
            # Find similar documents
            similar_docs = builder.query_hnsw(query_vector, k=5)
            
            print("Most similar document categories:")
            for i, category in enumerate(similar_docs):
                print(f"  {i+1}. {category}")
            
            print("\nThis demonstrates how to use HNSW for:")
            print("- Document similarity search")
            print("- Recommendation systems")
            print("- Image similarity")
            print("- Any vector similarity task")
            
    except Exception as e:
        print(f"Error in real-world example: {e}")

if __name__ == "__main__":
    main()
    create_real_world_example()
