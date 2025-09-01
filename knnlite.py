"""
Python bindings for KNNLite HNSW implementation

This module provides Python bindings for the FORTRAN HNSW implementation
by calling the compiled FORTRAN executable and parsing results.
"""

import numpy as np
import subprocess
import tempfile
import os
import json

class KNNClassifier:
    """
    HNSW-based k-Nearest Neighbor classifier with Python interface.
    
    This class provides a Python interface to the FORTRAN HNSW implementation
    by calling the compiled FORTRAN executable.
    """
    
    def __init__(self, dim=None, data=None, labels=None):
        """
        Initialize a new KNN classifier.
        
        Parameters:
        -----------
        dim : int, optional
            Dimension of vectors for classification
        data : array-like, optional
            Training data as a 2D array (n_samples, n_features)
        labels : array-like, optional
            Training labels as a 1D array of strings
            
        Returns:
        --------
        KNNClassifier
            A KNNClassifier instance
        """
        self._initialized = False
        self._dim = dim
        self._data = None
        self._labels = None
        
        if data is not None and labels is not None and dim is not None:
            self.fit(data, labels)
    
    def fit(self, data, labels):
        """
        Fit the HNSW classifier to training data.
        
        Parameters:
        -----------
        data : array-like
            Training data as a 2D array (n_samples, n_features)
        labels : array-like
            Training labels as a 1D array of strings
        """
        # Convert inputs to numpy arrays
        self._data = np.asarray(data, dtype=np.float32)
        self._labels = np.asarray(labels, dtype='U32')  # 32-character strings
        
        if self._data.ndim != 2:
            raise ValueError("Data must be a 2D array")
        
        n_samples, n_features = self._data.shape
        
        if len(self._labels) != n_samples:
            raise ValueError("Number of labels must match number of samples")
        
        self._dim = n_features
        self._initialized = True
    
    def predict(self, X, k=5):
        """
        Predict class labels for test vectors.
        
        Parameters:
        -----------
        X : array-like
            Test vectors as a 2D array (n_queries, n_features)
        k : int, default=5
            Number of nearest neighbors to consider
            
        Returns:
        --------
        predictions : array
            Predicted class labels
        """
        if not self._initialized:
            raise RuntimeError("Classifier not fitted. Call fit() first.")
        
        # Convert inputs to numpy arrays
        X = np.asarray(X, dtype=np.float32)
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        if X.shape[1] != self._dim:
            raise ValueError(f"Test vectors must have {self._dim} features")
        
        # Create a simple test program that uses our data
        test_program = f"""
program test_classify
    use knnlite
    implicit none
    
    integer, parameter :: dim = {self._dim}
    integer, parameter :: num_data = {len(self._data)}
    integer, parameter :: num_queries = {len(X)}
    integer, parameter :: k = {k}
    
    real :: data(num_data, dim)
    character(len=32) :: labels(num_data)
    real :: query_vectors(num_queries, dim)
    character(len=32), allocatable :: predictions(:)
    type(KNNClassifier) :: classifier
    
    integer :: i, j
    
    ! Initialize data
"""
        
        # Add data initialization
        for i, (point, label) in enumerate(zip(self._data, self._labels)):
            test_program += f"    data({i+1}, :) = ["
            test_program += ", ".join(map(str, point))
            test_program += f"]\n"
            test_program += f"    labels({i+1}) = '{label}'\n"
        
        # Add query vectors
        for i, point in enumerate(X):
            test_program += f"    query_vectors({i+1}, :) = ["
            test_program += ", ".join(map(str, point))
            test_program += f"]\n"
        
        test_program += """
    ! Create classifier
    classifier = NewKNNClassifier(dim, data, labels)
    
    ! Classify
    predictions = Classify(classifier, query_vectors, k)
    
    ! Print results
    do i = 1, num_queries
        print *, trim(predictions(i))
    end do
    
    deallocate(predictions)
    deallocate(classifier%nodes)
    
end program test_classify
"""
        
        # Write test program to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.f90', delete=False) as f:
            f.write(test_program)
            test_file = f.name
        
        try:
            # Compile and run the test program
            compile_result = subprocess.run(
                ['gfortran', '-O3', '-o', 'temp_test', 'knnlite.f90', test_file],
                capture_output=True,
                text=True
            )
            
            if compile_result.returncode != 0:
                raise RuntimeError(f"Compilation failed: {compile_result.stderr}")
            
            # Run the test program
            result = subprocess.run(
                ['./temp_test'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Execution failed: {result.stderr}")
            
            # Parse predictions from output
            predictions = result.stdout.strip().split('\n')
            predictions = [pred.strip() for pred in predictions if pred.strip()]
            
            if len(predictions) != len(X):
                raise RuntimeError(f"Expected {len(X)} predictions, got {len(predictions)}")
            
            return np.array(predictions)
            
        finally:
            # Clean up temporary files
            if os.path.exists(test_file):
                os.unlink(test_file)
            if os.path.exists('temp_test'):
                os.unlink('temp_test')
    
    @property
    def dim(self):
        """Get the dimension of the feature vectors."""
        return self._dim
    
    @property
    def num_nodes(self):
        """Get the number of nodes in the HNSW graph."""
        return len(self._data) if self._data is not None else 0

# Convenience function to match the FORTRAN API
def NewKNNClassifier(dim, data, labels):
    """
    Create a new KNN classifier (matches FORTRAN API).
    
    Parameters:
    -----------
    dim : int
        Dimension of vectors for classification
    data : array-like
        Training data as a 2D array (n_samples, n_features)
    labels : array-like
        Training labels as a 1D array of strings
        
    Returns:
    --------
    KNNClassifier
        A fitted KNNClassifier instance
    """
    return KNNClassifier(dim=dim, data=data, labels=labels)

# Convenience function to match the FORTRAN API
def Classify(classifier, query_vectors, k=5):
    """
    Classify vectors using a KNN classifier (matches FORTRAN API).
    
    Parameters:
    -----------
    classifier : KNNClassifier
        A fitted KNNClassifier instance
    query_vectors : array-like
        Query vectors as a 2D array (n_queries, n_features)
    k : int, default=5
        Number of nearest neighbors to consider
        
    Returns:
    --------
    predictions : array
        Predicted class labels
    """
    return classifier.predict(query_vectors, k)

# Example usage and testing
if __name__ == "__main__":
    # Test the Python bindings
    print("Testing KNNLite Python bindings...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 10
    n_features = 2
    
    # Generate clustered data
    data = np.random.randn(n_samples, n_features).astype(np.float32)
    labels = np.array([f"Class{i%3+1}" for i in range(n_samples)], dtype='U32')
    
    # Add some structure
    data[:3] += [0, 0]
    data[3:6] += [5, 5]
    data[6:] += [10, 10]
    
    # Create classifier
    print("Creating classifier...")
    classifier = NewKNNClassifier(n_features, data, labels)
    print(f"Classifier created with {classifier.num_nodes} nodes")
    
    # Test predictions
    print("\nTesting predictions...")
    test_points = np.array([[0.5, 0.5], [5.5, 5.5], [10.5, 10.5]], dtype=np.float32)
    predictions = classifier.predict(test_points, k=3)
    
    for i, (point, pred) in enumerate(zip(test_points, predictions)):
        print(f"Query point {i+1}: {point} -> Predicted class: {pred}")
    
    print("\nTest completed successfully!")
