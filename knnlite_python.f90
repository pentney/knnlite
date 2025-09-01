subroutine init_classifier(dim, num_data, data, labels, info)
    use knnlite
    implicit none
    
    !f2py intent(in) :: dim, num_data, data, labels
    !f2py intent(out) :: info
    !f2py depend(num_data, dim) :: data
    !f2py depend(num_data) :: labels
    
    integer, intent(in) :: dim, num_data
    real, intent(in) :: data(num_data, dim)
    character(len=32), intent(in) :: labels(num_data)
    integer, intent(out) :: info
    
    type(KNNClassifier), save :: classifier
    logical, save :: initialized = .false.
    
    info = 0
    
    ! Clean up previous classifier if exists
    if (initialized) then
        deallocate(classifier%nodes)
    end if
    
    ! Create new classifier
    classifier = NewKNNClassifier(dim, data, labels)
    initialized = .true.
    
end subroutine init_classifier

subroutine classify_vectors(num_queries, dim, query_vectors, k, predictions, info)
    use knnlite
    implicit none
    
    !f2py intent(in) :: num_queries, dim, query_vectors, k
    !f2py intent(out) :: predictions, info
    !f2py depend(num_queries, dim) :: query_vectors
    !f2py depend(num_queries) :: predictions
    
    integer, intent(in) :: num_queries, dim, k
    real, intent(in) :: query_vectors(num_queries, dim)
    character(len=32), intent(out) :: predictions(num_queries)
    integer, intent(out) :: info
    
    type(KNNClassifier), save :: classifier
    logical, save :: initialized = .false.
    character(len=32), allocatable :: predictions_temp(:)
    
    info = 0
    
    if (.not. initialized) then
        info = -1
        return
    end if
    
    ! Classify
    predictions_temp = Classify(classifier, query_vectors, k)
    
    ! Copy results
    predictions = predictions_temp
    
    deallocate(predictions_temp)
    
end subroutine classify_vectors

subroutine get_info(dim, num_nodes, max_layer, entry_point, info)
    !f2py intent(out) :: dim, num_nodes, max_layer, entry_point, info
    
    integer, intent(out) :: dim, num_nodes, max_layer, entry_point, info
    
    type(KNNClassifier), save :: classifier
    logical, save :: initialized = .false.
    
    info = 0
    
    if (.not. initialized) then
        info = -1
        return
    end if
    
    dim = classifier%dim
    num_nodes = classifier%num_nodes
    max_layer = classifier%max_layer
    entry_point = classifier%entry_point_index
    
end subroutine get_info

subroutine cleanup(info)
    !f2py intent(out) :: info
    
    integer, intent(out) :: info
    
    type(KNNClassifier), save :: classifier
    logical, save :: initialized = .false.
    
    info = 0
    
    if (initialized) then
        deallocate(classifier%nodes)
        initialized = .false.
    end if
    
end subroutine cleanup
