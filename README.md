# KNNLite - HNSW Implementation in FORTRAN

This is a FORTRAN implementation of the Hierarchical Navigable Small World (HNSW) algorithm for k-nearest neighbor classification, following the API structure described in `KNNLITE_API.md`.

## Features

- **HNSW Algorithm**: Implements the hierarchical navigable small world graph structure for efficient approximate nearest neighbor search
- **K-NN Classification**: Supports k-nearest neighbor classification with configurable k
- **FORTRAN 2008**: Modern FORTRAN implementation with derived types and allocatable arrays
- **API Compatibility**: Follows the structure described in the API documentation

## Files

- `knnlite.f90`: Main module containing the HNSW implementation
- `test_knnlite.f90`: Test program demonstrating usage
- `Makefile`: Build configuration
- `KNNLITE_API.md`: API documentation

## API Methods

### NewKNNClassifier(dim, data, labels)
Creates a new HNSW-based k-NN classifier.

**Input:**
- `dim`: Dimension of vectors for classification
- `data`: 2D array of n-dimensional float vectors (num_samples × dim)
- `labels`: Array of string labels corresponding to each data point

**Returns:**
- `KNNClassifier`: HNSW structure representing the data points for k-NN classification

### Classify(classifier, query_vectors, k)
Classifies one or more n-dimensional vectors using the HNSW classifier.

**Input:**
- `classifier`: KNNClassifier instance
- `query_vectors`: 2D array of query vectors (num_queries × dim)
- `k`: Number of nearest neighbors to consider

**Returns:**
- Array of predicted labels for each query vector

## HNSW Parameters

- `MAX_LAYERS = 16`: Maximum number of layers in the hierarchical structure
- `M = 16`: Maximum connections per node in upper layers
- `M_MAX = 32`: Maximum connections per node in layer 0
- `EF_CONSTRUCTION = 200`: Size of dynamic candidate list during construction
- `EF_SEARCH = 50`: Size of dynamic candidate list during search

## Building and Running

```bash
# Compile the code
make

# Run the test
make test

# Clean up
make clean
```

## Example Usage

```fortran
program example
    use knnlite
    implicit none
    
    integer, parameter :: dim = 2
    real :: data(10, dim)
    character(len=32) :: labels(10)
    real :: query_vectors(3, dim)
    character(len=32), allocatable :: predictions(:)
    type(KNNClassifier) :: classifier
    
    ! Initialize data and labels
    ! ... (setup your data)
    
    ! Create classifier
    classifier = NewKNNClassifier(dim, data, labels)
    
    ! Classify query points
    predictions = Classify(classifier, query_vectors, k=3)
    
    ! Use predictions
    ! ...
end program example
```

## Algorithm Details

The HNSW implementation includes:

1. **Hierarchical Structure**: Multi-layer graph where each node is assigned to layers using a geometric distribution
2. **Graph Construction**: Nodes are connected to their nearest neighbors at each layer
3. **Search Algorithm**: Greedy search starting from the top layer and moving down
4. **Distance Calculation**: Euclidean distance for vector similarity
5. **Classification**: Majority voting among k nearest neighbors

## Performance Notes

This is a simplified implementation for educational purposes. For production use, consider:
- Optimizing the search algorithm with proper candidate list management
- Implementing more sophisticated neighbor selection heuristics
- Adding support for different distance metrics
- Memory optimization for large datasets

## Dependencies

- FORTRAN 2008 compatible compiler (gfortran recommended)
- No external libraries required

## Database Integration

KNNLite now includes comprehensive PostgreSQL integration with pgvector support:

### Features
- **PostgreSQL Integration**: Read vector data directly from database tables
- **pgvector Support**: Full support for pgvector data types
- **Flexible Queries**: Support for WHERE clauses and data filtering
- **High Performance**: Leverages the fast FORTRAN HNSW implementation

### Quick Start

```python
from knnlite_database import DatabaseHNSWBuilder

# Database connection
db_params = {
    'host': 'localhost',
    'port': 5432,
    'database': 'your_database',
    'user': 'your_user',
    'password': 'your_password'
}

# Build HNSW index from database
with DatabaseHNSWBuilder(db_params) as builder:
    classifier = builder.build_hnsw_index(
        table_name='documents',
        vector_column='embedding',  # pgvector column
        label_column='category'     # classification column
    )
    
    # Query the index
    predictions = builder.query_hnsw(query_vectors, k=5)
```

### Setup

```bash
# Install database dependencies
pip install -r requirements_database.txt

# Setup database environment
python3 setup_database.py

# Run example
python3 database_example.py
```

For detailed database integration documentation, see [DATABASE_README.md](DATABASE_README.md).

