program example_usage
    use knnlite
    implicit none
    
    ! Example with more realistic data
    integer, parameter :: dim = 3
    integer, parameter :: num_samples = 20
    integer, parameter :: num_queries = 5
    integer, parameter :: k = 5
    
    real :: data(num_samples, dim)
    character(len=32) :: labels(num_samples)
    real :: query_vectors(num_queries, dim)
    character(len=32), allocatable :: predictions(:)
    type(KNNClassifier) :: classifier
    
    integer :: i
    
    ! Initialize random seed
    call random_seed()
    
    ! Create sample data (3D points with 3 classes)
    ! Class A: points around (1,1,1)
    data(1,:) = [1.0, 1.0, 1.0]; labels(1) = 'ClassA'
    data(2,:) = [1.1, 1.1, 1.1]; labels(2) = 'ClassA'
    data(3,:) = [0.9, 1.0, 1.1]; labels(3) = 'ClassA'
    data(4,:) = [1.2, 0.8, 1.0]; labels(4) = 'ClassA'
    data(5,:) = [0.8, 1.2, 0.9]; labels(5) = 'ClassA'
    data(6,:) = [1.0, 1.0, 0.8]; labels(6) = 'ClassA'
    
    ! Class B: points around (5,5,5)
    data(7,:) = [5.0, 5.0, 5.0]; labels(7) = 'ClassB'
    data(8,:) = [5.1, 5.1, 5.1]; labels(8) = 'ClassB'
    data(9,:) = [4.9, 5.0, 5.1]; labels(9) = 'ClassB'
    data(10,:) = [5.2, 4.8, 5.0]; labels(10) = 'ClassB'
    data(11,:) = [4.8, 5.2, 4.9]; labels(11) = 'ClassB'
    data(12,:) = [5.0, 5.0, 4.8]; labels(12) = 'ClassB'
    data(13,:) = [5.1, 4.9, 5.1]; labels(13) = 'ClassB'
    
    ! Class C: points around (9,9,9)
    data(14,:) = [9.0, 9.0, 9.0]; labels(14) = 'ClassC'
    data(15,:) = [9.1, 9.1, 9.1]; labels(15) = 'ClassC'
    data(16,:) = [8.9, 9.0, 9.1]; labels(16) = 'ClassC'
    data(17,:) = [9.2, 8.8, 9.0]; labels(17) = 'ClassC'
    data(18,:) = [8.8, 9.2, 8.9]; labels(18) = 'ClassC'
    data(19,:) = [9.0, 9.0, 8.8]; labels(19) = 'ClassC'
    data(20,:) = [9.1, 8.9, 9.1]; labels(20) = 'ClassC'
    
    ! Create query points
    query_vectors(1,:) = [1.5, 1.5, 1.5]  ! Should be close to ClassA
    query_vectors(2,:) = [5.5, 5.5, 5.5]  ! Should be close to ClassB
    query_vectors(3,:) = [8.5, 8.5, 8.5]  ! Should be close to ClassC
    query_vectors(4,:) = [3.0, 3.0, 3.0]  ! Between classes, should be interesting
    query_vectors(5,:) = [7.0, 7.0, 7.0]  ! Between classes, should be interesting
    
    ! Create the HNSW classifier
    print *, '=== KNNLite HNSW Example ==='
    print *, 'Creating HNSW classifier with', num_samples, 'samples of dimension', dim
    classifier = NewKNNClassifier(dim, data, labels)
    print *, 'Classifier created successfully!'
    print *, 'Number of nodes:', classifier%num_nodes
    print *, 'Maximum layer:', classifier%max_layer
    print *, 'Entry point index:', classifier%entry_point_index
    print *
    
    ! Classify query points
    print *, 'Classifying', num_queries, 'query points with k =', k
    predictions = Classify(classifier, query_vectors, k)
    print *
    
    ! Print results
    print *, 'Classification Results:'
    print *, '======================'
    do i = 1, num_queries
        print '(A,I0,A,3F8.2,A,A)', 'Query ', i, ': [', query_vectors(i,:), '] -> ', trim(predictions(i))
    end do
    print *
    
    ! Show some statistics about the HNSW structure
    print *, 'HNSW Structure Statistics:'
    print *, '=========================='
    do i = 1, min(5, classifier%num_nodes)  ! Show first 5 nodes
        print '(A,I0,A,I0,A,A)', 'Node ', i, ' (layer ', classifier%nodes(i)%layer, '): ', trim(classifier%nodes(i)%label)
    end do
    if (classifier%num_nodes > 5) then
        print *, '... and', classifier%num_nodes - 5, 'more nodes'
    end if
    
    print *
    print *, 'Example completed successfully!'
    
end program example_usage
