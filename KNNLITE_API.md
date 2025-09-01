Methods for KNNLite:

NewKNNClassifier() - method that creates a ball tree for k-NN modeling
Input:
    - dim - dimension of vectors for classification
    - data - a list of n-dimensional float vectors and string labels
Returns:
    - a KNNClassifier model (JSON-serializable) representing the data points for the kNN classifier

KNNClassifier - data for point classification (represented as a ball tree)

Classify() - method to classify one or more n-dimensional vectors with a given KNNClassifier
