"""
KNNLite Database Integration Module

This module provides functionality to read vector data from PostgreSQL tables
with pgvector support and build HNSW indices for k-nearest neighbor classification.
"""

import numpy as np
import psycopg2
import psycopg2.extras
from typing import List, Tuple, Optional, Union
import logging
from knnlite_python import KNNClassifier, NewKNNClassifier

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseHNSWBuilder:
    """
    Builds HNSW indices from PostgreSQL tables with pgvector support.
    """
    
    def __init__(self, connection_params: dict):
        """
        Initialize the database HNSW builder.
        
        Parameters:
        -----------
        connection_params : dict
            PostgreSQL connection parameters including:
            - host: Database host
            - port: Database port (default: 5432)
            - database: Database name
            - user: Username
            - password: Password
        """
        self.connection_params = connection_params
        self.connection = None
        self.classifier = None
        
    def connect(self):
        """Establish database connection."""
        try:
            self.connection = psycopg2.connect(**self.connection_params)
            logger.info("Successfully connected to PostgreSQL database")
        except psycopg2.Error as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def disconnect(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
    
    def read_vector_data(self, 
                        table_name: str, 
                        vector_column: str, 
                        label_column: str,
                        where_clause: Optional[str] = None,
                        limit: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read vector data and labels from a PostgreSQL table.
        
        Parameters:
        -----------
        table_name : str
            Name of the table containing the data
        vector_column : str
            Name of the column containing pgvector data
        label_column : str
            Name of the column containing classification labels
        where_clause : str, optional
            SQL WHERE clause to filter data (without 'WHERE' keyword)
        limit : int, optional
            Maximum number of rows to read
            
        Returns:
        --------
        vectors : np.ndarray
            Array of vectors (n_samples, n_features)
        labels : np.ndarray
            Array of labels (n_samples,)
        """
        if not self.connection:
            raise RuntimeError("Not connected to database. Call connect() first.")
        
        # Build the query
        query = f"""
        SELECT {vector_column}, {label_column}
        FROM {table_name}
        """
        
        if where_clause:
            query += f" WHERE {where_clause}"
        
        if limit:
            query += f" LIMIT {limit}"
        
        logger.info(f"Executing query: {query}")
        
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()
                
                if not rows:
                    raise ValueError("No data found in the specified table/conditions")
                
                # Extract vectors and labels
                vectors = []
                labels = []
                
                for row in rows:
                    vector_data, label = row
                    
                    # Handle pgvector data
                    if isinstance(vector_data, str):
                        # Parse pgvector string format: [1,2,3] or (1,2,3)
                        vector_str = vector_data.strip('[]()')
                        vector = np.array([float(x.strip()) for x in vector_str.split(',')])
                    elif hasattr(vector_data, '__iter__'):
                        # Handle array-like data
                        vector = np.array(vector_data, dtype=np.float32)
                    else:
                        raise ValueError(f"Unsupported vector format: {type(vector_data)}")
                    
                    vectors.append(vector)
                    labels.append(str(label))
                
                vectors = np.array(vectors, dtype=np.float32)
                labels = np.array(labels, dtype='U32')
                
                logger.info(f"Loaded {len(vectors)} vectors with dimension {vectors.shape[1]}")
                return vectors, labels
                
        except psycopg2.Error as e:
            logger.error(f"Database error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            raise
    
    def build_hnsw_index(self, 
                        table_name: str, 
                        vector_column: str, 
                        label_column: str,
                        where_clause: Optional[str] = None,
                        limit: Optional[int] = None,
                        **kwargs) -> KNNClassifier:
        """
        Build an HNSW index from database data.
        
        Parameters:
        -----------
        table_name : str
            Name of the table containing the data
        vector_column : str
            Name of the column containing pgvector data
        label_column : str
            Name of the column containing classification labels
        where_clause : str, optional
            SQL WHERE clause to filter data
        limit : int, optional
            Maximum number of rows to read
        **kwargs
            Additional arguments passed to NewKNNClassifier
            
        Returns:
        --------
        classifier : KNNClassifier
            Trained HNSW classifier
        """
        logger.info(f"Building HNSW index from table '{table_name}'")
        
        # Read data from database
        vectors, labels = self.read_vector_data(
            table_name, vector_column, label_column, where_clause, limit
        )
        
        # Build HNSW classifier
        logger.info("Training HNSW classifier...")
        self.classifier = NewKNNClassifier(vectors.shape[1], vectors, labels, **kwargs)
        
        logger.info(f"HNSW index built successfully with {self.classifier.num_nodes} nodes")
        return self.classifier
    
    def query_hnsw(self, 
                   query_vectors: np.ndarray, 
                   k: int = 5) -> np.ndarray:
        """
        Query the HNSW index for nearest neighbors.
        
        Parameters:
        -----------
        query_vectors : np.ndarray
            Query vectors (n_queries, n_features)
        k : int
            Number of nearest neighbors to return
            
        Returns:
        --------
        predictions : np.ndarray
            Predicted labels for each query vector
        """
        if self.classifier is None:
            raise RuntimeError("No HNSW index built. Call build_hnsw_index() first.")
        
        return self.classifier.predict(query_vectors, k)
    
    def get_index_info(self) -> dict:
        """
        Get information about the built HNSW index.
        
        Returns:
        --------
        info : dict
            Dictionary containing index information
        """
        if self.classifier is None:
            raise RuntimeError("No HNSW index built. Call build_hnsw_index() first.")
        
        return {
            'dimension': self.classifier.dim,
            'num_nodes': self.classifier.num_nodes,
            'max_layer': getattr(self.classifier, 'max_layer', 'N/A'),
            'entry_point': getattr(self.classifier, 'entry_point', 'N/A')
        }

def create_sample_table(connection_params: dict, 
                       table_name: str = "sample_vectors",
                       num_samples: int = 1000,
                       dimension: int = 128):
    """
    Create a sample table with vector data for testing.
    
    Parameters:
    -----------
    connection_params : dict
        PostgreSQL connection parameters
    table_name : str
        Name of the table to create
    num_samples : int
        Number of sample vectors to generate
    dimension : int
        Dimension of the vectors
    """
    with psycopg2.connect(**connection_params) as conn:
        with conn.cursor() as cursor:
            # Create table with pgvector extension
            cursor.execute(f"""
                CREATE EXTENSION IF NOT EXISTS vector;
            """)
            
            cursor.execute(f"""
                DROP TABLE IF EXISTS {table_name};
            """)
            
            cursor.execute(f"""
                CREATE TABLE {table_name} (
                    id SERIAL PRIMARY KEY,
                    vector vector({dimension}),
                    label VARCHAR(50)
                );
            """)
            
            # Generate sample data
            logger.info(f"Generating {num_samples} sample vectors...")
            
            # Create clusters
            num_clusters = 5
            cluster_centers = np.random.randn(num_clusters, dimension) * 10
            
            for i in range(num_samples):
                cluster_id = i % num_clusters
                # Generate vector near cluster center with some noise
                vector = cluster_centers[cluster_id] + np.random.randn(dimension) * 2
                label = f"cluster_{cluster_id + 1}"
                
                # Convert to pgvector format
                vector_str = '[' + ','.join(map(str, vector)) + ']'
                
                cursor.execute(f"""
                    INSERT INTO {table_name} (vector, label)
                    VALUES (%s::vector, %s);
                """, (vector_str, label))
            
            conn.commit()
            logger.info(f"Created table '{table_name}' with {num_samples} vectors")

def benchmark_database_hnsw(connection_params: dict,
                           table_name: str = "sample_vectors",
                           test_sizes: List[int] = [1000, 5000, 10000],
                           test_dimensions: List[int] = [64, 128, 256]):
    """
    Benchmark HNSW performance on database data.
    
    Parameters:
    -----------
    connection_params : dict
        PostgreSQL connection parameters
    table_name : str
        Name of the table to benchmark
    test_sizes : List[int]
        List of sample sizes to test
    test_dimensions : List[int]
        List of vector dimensions to test
    """
    print("=" * 60)
    print("KNNLite Database HNSW Benchmark")
    print("=" * 60)
    
    with DatabaseHNSWBuilder(connection_params) as builder:
        for dim in test_dimensions:
            print(f"\nTesting dimension {dim}:")
            print("-" * 40)
            
            for size in test_sizes:
                try:
                    # Build index
                    start_time = time.time()
                    classifier = builder.build_hnsw_index(
                        table_name, 'vector', 'label', 
                        limit=size
                    )
                    build_time = time.time() - start_time
                    
                    # Test queries
                    query_vectors = np.random.randn(100, dim).astype(np.float32)
                    start_time = time.time()
                    predictions = builder.query_hnsw(query_vectors, k=5)
                    query_time = time.time() - start_time
                    
                    print(f"Size {size:5d}: Build {build_time:.3f}s, "
                          f"Query {query_time:.3f}s, "
                          f"Rate {size/build_time:.0f} pts/s")
                    
                except Exception as e:
                    print(f"Size {size:5d}: Error - {e}")

# Example usage
if __name__ == "__main__":
    import time
    
    # Database connection parameters
    db_params = {
        'host': 'localhost',
        'port': 5432,
        'database': 'testdb',
        'user': 'postgres',
        'password': 'password'
    }
    
    try:
        # Create sample table
        print("Creating sample table...")
        create_sample_table(db_params, 'test_vectors', 5000, 128)
        
        # Build HNSW index
        print("\nBuilding HNSW index...")
        with DatabaseHNSWBuilder(db_params) as builder:
            classifier = builder.build_hnsw_index(
                'test_vectors', 'vector', 'label', limit=1000
            )
            
            # Get index info
            info = builder.get_index_info()
            print(f"Index info: {info}")
            
            # Test queries
            print("\nTesting queries...")
            test_vectors = np.random.randn(5, 128).astype(np.float32)
            predictions = builder.query_hnsw(test_vectors, k=3)
            
            for i, pred in enumerate(predictions):
                print(f"Query {i+1}: {pred}")
        
        print("\nExample completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure PostgreSQL is running and pgvector extension is installed.")
