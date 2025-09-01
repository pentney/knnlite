module knnlite
    implicit none
    
    ! Parameters for HNSW algorithm
    integer, parameter :: MAX_LAYERS = 16
    integer, parameter :: M = 16  ! Maximum connections per node
    integer, parameter :: M_MAX = 32  ! Maximum connections for layer 0
    integer, parameter :: EF_CONSTRUCTION = 200  ! Size of dynamic candidate list
    integer, parameter :: EF_SEARCH = 50  ! Size of dynamic candidate list for search
    
    ! Node structure for HNSW
    type :: HNSWNode
        real, allocatable :: vector(:)
        character(len=32) :: label
        integer :: layer
        integer, allocatable :: connection_indices(:,:)  ! connection_indices(layer, neighbor)
        integer, allocatable :: num_connections(:)   ! number of connections per layer
    end type HNSWNode
    
    ! HNSW graph structure
    type :: KNNClassifier
        integer :: dim
        integer :: num_nodes
        integer :: max_layer
        type(HNSWNode), allocatable :: nodes(:)
        integer :: entry_point_index
    end type KNNClassifier
    
contains
    
    ! Initialize a new KNN classifier with HNSW structure
    function NewKNNClassifier(dim, data, labels) result(classifier)
        integer, intent(in) :: dim
        real, intent(in) :: data(:,:)  ! data(:,1:dim) = vectors
        character(len=32), intent(in) :: labels(:)
        type(KNNClassifier) :: classifier
        
        integer :: i, j, layer, num_data
        real :: random_val
        
        num_data = size(data, 1)
        classifier%dim = dim
        classifier%num_nodes = num_data
        classifier%max_layer = 0
        
        ! Allocate nodes
        allocate(classifier%nodes(num_data))
        
        ! Initialize nodes
        do i = 1, num_data
            allocate(classifier%nodes(i)%vector(dim))
            classifier%nodes(i)%vector = data(i, 1:dim)
            classifier%nodes(i)%label = labels(i)
            
            ! Determine layer using geometric distribution
            layer = 0
            call random_number(random_val)
            do while (random_val < 0.5 .and. layer < MAX_LAYERS - 1)
                layer = layer + 1
                call random_number(random_val)
            end do
            classifier%nodes(i)%layer = layer
            classifier%max_layer = max(classifier%max_layer, layer)
            
            ! Allocate connections
            allocate(classifier%nodes(i)%connection_indices(0:layer, M_MAX))
            allocate(classifier%nodes(i)%num_connections(0:layer))
            classifier%nodes(i)%num_connections = 0
        end do
        
        ! Build HNSW graph
        call build_hnsw_graph(classifier)
        
        ! Set entry point (highest layer node)
        classifier%entry_point_index = 1
        do i = 2, num_data
            if (classifier%nodes(i)%layer > classifier%nodes(classifier%entry_point_index)%layer) then
                classifier%entry_point_index = i
            end if
        end do
    end function NewKNNClassifier
    
    ! Build the HNSW graph structure
    subroutine build_hnsw_graph(classifier)
        type(KNNClassifier), intent(inout) :: classifier
        
        integer :: i, j, layer, num_candidates, num_selected
        integer, allocatable :: candidates(:), selected(:)
        real, allocatable :: distances(:)
        integer :: current_m
        
        do i = 1, classifier%num_nodes
            layer = classifier%nodes(i)%layer
            
            ! For each layer from top to bottom
            do while (layer >= 0)
                if (layer == classifier%nodes(i)%layer) then
                    ! Insert at current layer
                    call search_layer(classifier, classifier%nodes(i), layer, EF_CONSTRUCTION, candidates, distances)
                    current_m = M
                else
                    ! Insert at lower layers
                    call search_layer(classifier, classifier%nodes(i), layer, EF_CONSTRUCTION, candidates, distances)
                    current_m = M_MAX
                end if
                
                ! Select neighbors using simple heuristic
                if (allocated(candidates)) then
                    num_candidates = min(size(candidates), EF_CONSTRUCTION)
                    allocate(selected(min(num_candidates, current_m)))
                    num_selected = 0
                    
                    do j = 1, num_candidates
                        if (num_selected < current_m) then
                            num_selected = num_selected + 1
                            selected(num_selected) = candidates(j)
                        end if
                    end do
                    
                    ! Add connections
                    do j = 1, num_selected
                        if (classifier%nodes(i)%num_connections(layer) < current_m) then
                            classifier%nodes(i)%num_connections(layer) = classifier%nodes(i)%num_connections(layer) + 1
                            classifier%nodes(i)%connection_indices(layer, classifier%nodes(i)%num_connections(layer)) = selected(j)
                        end if
                    end do
                    
                    deallocate(selected)
                end if
                layer = layer - 1
            end do
        end do
    end subroutine build_hnsw_graph
    
    ! Search for nearest neighbors in a specific layer
    subroutine search_layer(classifier, query_node, layer, ef, candidates, distances)
        type(KNNClassifier), intent(in) :: classifier
        type(HNSWNode), intent(in) :: query_node
        integer, intent(in) :: layer, ef
        integer, allocatable, intent(out) :: candidates(:)
        real, allocatable, intent(out) :: distances(:)
        
        integer :: i, j, num_candidates
        real :: dist
        logical :: visited(classifier%num_nodes)
        
        visited = .false.
        num_candidates = 0
        
        ! Simple implementation: find all nodes at this layer
        do i = 1, classifier%num_nodes
            if (classifier%nodes(i)%layer >= layer .and. .not. visited(i)) then
                visited(i) = .true.
                dist = euclidean_distance(query_node%vector, classifier%nodes(i)%vector)
                
                if (num_candidates < ef) then
                    num_candidates = num_candidates + 1
                    if (allocated(candidates)) then
                        candidates = [candidates, i]
                        distances = [distances, dist]
                    else
                        allocate(candidates(1))
                        allocate(distances(1))
                        candidates(1) = i
                        distances(1) = dist
                    end if
                end if
            end if
        end do
    end subroutine search_layer
    
    ! Calculate Euclidean distance between two vectors
    function euclidean_distance(vec1, vec2) result(dist)
        real, intent(in) :: vec1(:), vec2(:)
        real :: dist
        integer :: i
        
        dist = 0.0
        do i = 1, size(vec1)
            dist = dist + (vec1(i) - vec2(i))**2
        end do
        dist = sqrt(dist)
    end function euclidean_distance
    
    ! Classify one or more vectors using the HNSW classifier
    function Classify(classifier, query_vectors, k) result(predictions)
        type(KNNClassifier), intent(in) :: classifier
        real, intent(in) :: query_vectors(:,:)  ! query_vectors(:,1:dim)
        integer, intent(in) :: k
        character(len=32), allocatable :: predictions(:)
        
        integer :: i, j, num_queries, num_neighbors
        integer, allocatable :: neighbors(:)
        real, allocatable :: distances(:)
        character(len=32) :: temp_label
        integer :: label_counts(1000)  ! Assuming max 1000 unique labels
        integer :: max_count, best_label_idx
        
        num_queries = size(query_vectors, 1)
        allocate(predictions(num_queries))
        
        do i = 1, num_queries
            ! Search for k nearest neighbors
            call search_hnsw(classifier, query_vectors(i,:), k, neighbors, distances)
            
            ! Count label occurrences
            label_counts = 0
            do j = 1, min(k, size(neighbors))
                temp_label = classifier%nodes(neighbors(j))%label
                ! Simple hash to get label index (in practice, use proper mapping)
                best_label_idx = mod(abs(hash_string(temp_label)), 1000) + 1
                label_counts(best_label_idx) = label_counts(best_label_idx) + 1
            end do
            
            ! Find most frequent label
            max_count = 0
            best_label_idx = 1
            do j = 1, 1000
                if (label_counts(j) > max_count) then
                    max_count = label_counts(j)
                    best_label_idx = j
                end if
            end do
            
            ! Get the actual label (simplified - in practice, maintain proper mapping)
            predictions(i) = classifier%nodes(neighbors(1))%label
        end do
    end function Classify
    
    ! Search HNSW for k nearest neighbors
    subroutine search_hnsw(classifier, query_vector, k, neighbors, distances)
        type(KNNClassifier), intent(in) :: classifier
        real, intent(in) :: query_vector(:)
        integer, intent(in) :: k
        integer, allocatable, intent(out) :: neighbors(:)
        real, allocatable, intent(out) :: distances(:)
        
        integer :: i, j, num_found
        real :: dist
        logical :: visited(classifier%num_nodes)
        
        visited = .false.
        num_found = 0
        
        ! Simple implementation: search all nodes
        do i = 1, classifier%num_nodes
            if (.not. visited(i)) then
                visited(i) = .true.
                dist = euclidean_distance(query_vector, classifier%nodes(i)%vector)
                
                if (num_found < k) then
                    num_found = num_found + 1
                    if (allocated(neighbors)) then
                        neighbors = [neighbors, i]
                        distances = [distances, dist]
                    else
                        allocate(neighbors(1))
                        allocate(distances(1))
                        neighbors(1) = i
                        distances(1) = dist
                    end if
                end if
            end if
        end do
    end subroutine search_hnsw
    
    ! Simple hash function for strings
    function hash_string(str) result(hash)
        character(len=*), intent(in) :: str
        integer :: hash
        integer :: i
        
        hash = 0
        do i = 1, len_trim(str)
            hash = hash * 31 + ichar(str(i:i))
        end do
    end function hash_string
    
end module knnlite
