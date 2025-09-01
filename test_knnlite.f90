program test_knnlite
    use knnlite
    implicit none
    
    ! Test data
    integer, parameter :: dim = 2
    integer, parameter :: num_samples = 10
    integer, parameter :: num_queries = 3
    integer, parameter :: k = 3
    
    real :: data(num_samples, dim)
    character(len=32) :: labels(num_samples)
    real :: query_vectors(num_queries, dim)
    character(len=32), allocatable :: predictions(:)
    type(KNNClassifier) :: classifier
    
    integer :: i, j
    
    ! Initialize random seed
    call random_seed()
    
    ! Create sample data (2D points with labels)
    data(1,:) = [1.0, 1.0]; labels(1) = 'A'
    data(2,:) = [1.1, 1.1]; labels(2) = 'A'
    data(3,:) = [1.2, 1.0]; labels(3) = 'A'
    data(4,:) = [5.0, 5.0]; labels(4) = 'B'
    data(5,:) = [5.1, 5.1]; labels(5) = 'B'
    data(6,:) = [5.2, 5.0]; labels(6) = 'B'
    data(7,:) = [9.0, 9.0]; labels(7) = 'C'
    data(8,:) = [9.1, 9.1]; labels(8) = 'C'
    data(9,:) = [9.2, 9.0]; labels(9) = 'C'
    data(10,:) = [3.0, 3.0]; labels(10) = 'D'
    
    ! Create query points
    query_vectors(1,:) = [1.5, 1.5]  ! Should be close to cluster A
    query_vectors(2,:) = [5.5, 5.5]  ! Should be close to cluster B
    query_vectors(3,:) = [8.5, 8.5]  ! Should be close to cluster C
    
    ! Create the HNSW classifier
    print *, 'Creating HNSW classifier...'
    classifier = NewKNNClassifier(dim, data, labels)
    print *, 'Classifier created with', classifier%num_nodes, 'nodes'
    print *, 'Maximum layer:', classifier%max_layer
    
    ! Classify query points
    print *, 'Classifying query points...'
    predictions = Classify(classifier, query_vectors, k)
    
    ! Print results
    do i = 1, num_queries
        print *, 'Query point', i, ':', query_vectors(i,:), '-> Predicted class:', trim(predictions(i))
    end do
    
    print *, 'Test completed successfully!'
    
end program test_knnlite
