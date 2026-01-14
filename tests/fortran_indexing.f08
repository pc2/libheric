
PROGRAM indexing
    IMPLICIT NONE
    INTEGER, DIMENSION(:), ALLOCATABLE :: arr, ind_idx
    INTEGER, DIMENSION(:, :), ALLOCATABLE :: arr_off
    INTEGER, DIMENSION(:, :), ALLOCATABLE :: part
    INTEGER :: i

    CALL alloc_array_off(arr_off)
    WRITE (*, *) arr_off(0, 2)

    ALLOCATE(part(4, 2))

    ALLOCATE(arr(1000))
    ALLOCATE(ind_idx(5))

    ind_idx = (/ 2, 1, 3, 5, 4/)

    DO i = 1, SIZE(arr)
        arr(i) = i
    END DO

    WRITE (*, *) arr(ind_idx)
CONTAINS
    SUBROUTINE alloc_array_off(arr_off)
        IMPLICIT NONE
        INTEGER, DIMENSION(:, :), ALLOCATABLE, INTENT(OUT) :: arr_off
        ALLOCATE(arr_off(0:4, 2:3))
        arr_off(0, 2) = 21
    END SUBROUTINE
END PROGRAM

