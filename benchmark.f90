program benchmark
    use knnlite
    implicit none
    
    ! Benchmark parameters
    integer, parameter :: num_trials = 3
    integer, parameter :: k = 10
    integer, parameter :: num_queries = 500
    
    ! Test dimensions and sizes (reduced for faster testing)
    integer, dimension(4) :: test_dims = [2, 10, 50, 100]
    integer, dimension(3) :: test_sizes = [1000, 5000, 10000]
    
    ! Timing variables
    real :: start_time, end_time, build_time, search_time
    real :: total_build_time, total_search_time
    real :: avg_build_time, avg_search_time
    
    ! Data arrays
    real, allocatable :: data(:,:), query_vectors(:,:)
    character(len=32), allocatable :: labels(:)
    character(len=32), allocatable :: predictions(:)
    type(KNNClassifier) :: classifier
    
    ! Counters and variables
    integer :: trial, dim, size_idx, num_data
    real :: estimated_memory
    
    ! Random seed
    call random_seed()
    
    print *, '========================================'
    print *, 'KNNLite HNSW Benchmark Suite'
    print *, '========================================'
    print *, 'Testing HNSW model building and search performance'
    print *, 'Number of trials per test:', num_trials
    print *, 'K for classification:', k
    print *, 'Number of queries per test:', num_queries
    print *
    
    ! Benchmark 1: Model Building Performance
    print *, 'BENCHMARK 1: Model Building Performance'
    print *, '======================================='
    print *, 'Dimension | Data Size | Avg Build Time (s) | Build Rate (points/s)'
    print *, '----------|-----------|-------------------|---------------------'
    
    do dim = 1, size(test_dims)
        do size_idx = 1, size(test_sizes)
            num_data = test_sizes(size_idx)
            
            ! Generate random data
            allocate(data(num_data, test_dims(dim)))
            allocate(labels(num_data))
            call generate_random_data(data, labels, test_dims(dim), num_data)
            
            total_build_time = 0.0
            
            do trial = 1, num_trials
                call cpu_time(start_time)
                classifier = NewKNNClassifier(test_dims(dim), data, labels)
                call cpu_time(end_time)
                
                build_time = end_time - start_time
                total_build_time = total_build_time + build_time
                
                ! Clean up classifier for next trial
                deallocate(classifier%nodes)
            end do
            
            avg_build_time = total_build_time / real(num_trials)
            print '(I9,A,I10,A,F18.6,A,F20.0)', test_dims(dim), ' | ', num_data, ' | ', &
                  avg_build_time, ' | ', real(num_data) / avg_build_time
            
            deallocate(data, labels)
        end do
        print *
    end do
    
    ! Benchmark 2: Search Performance
    print *, 'BENCHMARK 2: Search Performance'
    print *, '=============================='
    print *, 'Dimension | Data Size | Avg Search Time (s) | Search Rate (queries/s)'
    print *, '----------|-----------|-------------------|----------------------'
    
    do dim = 1, size(test_dims)
        do size_idx = 1, size(test_sizes)
            num_data = test_sizes(size_idx)
            
            ! Generate random data
            allocate(data(num_data, test_dims(dim)))
            allocate(labels(num_data))
            call generate_random_data(data, labels, test_dims(dim), num_data)
            
            ! Build classifier once
            classifier = NewKNNClassifier(test_dims(dim), data, labels)
            
            ! Generate query vectors
            allocate(query_vectors(num_queries, test_dims(dim)))
            call generate_query_vectors(query_vectors, test_dims(dim), num_queries)
            
            total_search_time = 0.0
            
            do trial = 1, num_trials
                call cpu_time(start_time)
                predictions = Classify(classifier, query_vectors, k)
                call cpu_time(end_time)
                
                search_time = end_time - start_time
                total_search_time = total_search_time + search_time
                
                deallocate(predictions)
            end do
            
            avg_search_time = total_search_time / real(num_trials)
            print '(I9,A,I10,A,F18.6,A,F22.0)', test_dims(dim), ' | ', num_data, ' | ', &
                  avg_search_time, ' | ', real(num_queries) / avg_search_time
            
            deallocate(data, labels, query_vectors)
            deallocate(classifier%nodes)
        end do
        print *
    end do
    
    ! Benchmark 3: Accuracy Test
    print *, 'BENCHMARK 3: Classification Accuracy'
    print *, '==================================='
    print *, 'Testing accuracy on synthetic dataset with known clusters'
    print *
    
    call accuracy_benchmark()
    
    ! Benchmark 4: Memory Usage (approximate)
    print *, 'BENCHMARK 4: Memory Usage Estimation'
    print *, '==================================='
    print *, 'Dimension | Data Size | Est. Memory (MB) | Memory per Point (bytes)'
    print *, '----------|-----------|-----------------|------------------------'
    
    do dim = 1, size(test_dims)
        do size_idx = 1, size(test_sizes)
            num_data = test_sizes(size_idx)
            
            ! Estimate memory usage
            ! Each node: vector(dim) + label(32) + layer(4) + connections + num_connections
            estimated_memory = real(num_data) * (real(test_dims(dim)) * 4.0 + 32.0 + 4.0 + &
                          real(MAX_LAYERS * M_MAX * 4) + real(MAX_LAYERS * 4)) / (1024.0 * 1024.0)
            
            print '(I9,A,I10,A,F15.2,A,F25.1)', test_dims(dim), ' | ', num_data, ' | ', &
                  estimated_memory, ' | ', estimated_memory * 1024.0 * 1024.0 / real(num_data)
        end do
        print *
    end do
    
    print *, '========================================'
    print *, 'Benchmark completed successfully!'
    print *, '========================================'
    
contains
    
    ! Generate random data with some structure
    subroutine generate_random_data(data, labels, dim, num_data)
        real, intent(out) :: data(:,:)
        character(len=32), intent(out) :: labels(:)
        integer, intent(in) :: dim, num_data
        
        integer :: i, j, cluster
        real :: center(dim)
        real :: noise
        
        do i = 1, num_data
            ! Assign to one of 5 clusters
            cluster = mod(i-1, 5) + 1
            
            ! Set cluster center
            do j = 1, dim
                center(j) = real(cluster) * 2.0
            end do
            
            ! Generate point with some noise
            do j = 1, dim
                call random_number(noise)
                data(i, j) = center(j) + (noise - 0.5) * 0.5
            end do
            
            ! Set label
            write(labels(i), '(A,I0)') 'Cluster', cluster
        end do
    end subroutine generate_random_data
    
    ! Generate query vectors
    subroutine generate_query_vectors(query_vectors, dim, num_queries)
        real, intent(out) :: query_vectors(:,:)
        integer, intent(in) :: dim, num_queries
        
        integer :: i, j
        real :: val
        
        do i = 1, num_queries
            do j = 1, dim
                call random_number(val)
                query_vectors(i, j) = val * 10.0  ! Scale to [0, 10]
            end do
        end do
    end subroutine generate_query_vectors
    
    ! Test accuracy on synthetic dataset
    subroutine accuracy_benchmark()
        integer, parameter :: dim = 2
        integer, parameter :: num_data = 1000
        integer, parameter :: num_queries = 100
        integer, parameter :: k = 5
        
        real :: data(num_data, dim)
        character(len=32) :: labels(num_data)
        real :: query_vectors(num_queries, dim)
        character(len=32), allocatable :: predictions(:)
        type(KNNClassifier) :: classifier
        
        integer :: i, j, cluster
        real :: center(dim)
        character(len=32) :: true_label
        integer :: num_correct
        
        ! Generate structured data with clear clusters
        do i = 1, num_data
            cluster = mod(i-1, 3) + 1  ! 3 clusters
            
            ! Set cluster center
            center(1) = real(cluster) * 3.0
            center(2) = real(cluster) * 3.0
            
            ! Generate point with small noise
            do j = 1, dim
                call random_number(data(i, j))
                data(i, j) = center(j) + (data(i, j) - 0.5) * 0.3
            end do
            
            write(labels(i), '(A,I0)') 'Class', cluster
        end do
        
        ! Build classifier
        classifier = NewKNNClassifier(dim, data, labels)
        
        ! Generate test queries
        num_correct = 0
        do i = 1, num_queries
            cluster = mod(i-1, 3) + 1
            
            ! Generate query near cluster center
            center(1) = real(cluster) * 3.0
            center(2) = real(cluster) * 3.0
            
            do j = 1, dim
                call random_number(query_vectors(i, j))
                query_vectors(i, j) = center(j) + (query_vectors(i, j) - 0.5) * 0.5
            end do
            
            write(true_label, '(A,I0)') 'Class', cluster
            
            ! Classify
            predictions = Classify(classifier, query_vectors(i:i,:), k)
            
            ! Check accuracy
            if (trim(predictions(1)) == trim(true_label)) then
                num_correct = num_correct + 1
            end if
            
            deallocate(predictions)
        end do
        
        print *, 'Accuracy on synthetic dataset:', real(num_correct) / real(num_queries) * 100.0, '%'
        print *, 'Correct predictions:', num_correct, 'out of', num_queries
        print *
        
        deallocate(classifier%nodes)
    end subroutine accuracy_benchmark
    
end program benchmark
