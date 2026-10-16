PROGRAM fortran_interface_test
   USE libheric_interface, ONLY: query_accelerator_devices, &
                                 create_accelerator_context, &
                                 free_accelerator_context, &
                                 accelerator_context_functional, &
                                 accelerator_context_set_quartet, &
                                 accelerator_context_set_quartet_permutation, &
                                 accelerator_context_set_atom_id, &
                                 accelerator_context_set_set_id, &
                                 accelerator_context_set_shell_id, &
                                 accelerator_context_set_exponent_id, &
                                 accelerator_context_set_offset, &
                                 accelerator_context_submit_peri, &
                                 accelerator_context_submit_ceri, &
                                 accelerator_context_finish_quartet_class, &
                                 accelerator_context_get_next_ceri, &
                                 generate_quartet_permutations, &
                                 pc2_hide_acl, &
                                 acc_kind, acc_device, & 
                                 LIBHERIC_MAX_AM, &
                                 LIBHERIC_NSO
   USE mpi_f08,            ONLY: MPI_Init, MPI_Finalize, MPI_Comm_rank, &
                                 MPI_Comm_size, MPI_COMM_WORLD, MPI_Barrier 
   USE ISO_C_BINDING,      ONLY: C_DOUBLE, C_PTR, C_NULL_CHAR
   USE OMP_LIB,            ONLY: OMP_get_thread_num, OMP_set_num_threads
   
   IMPLICIT NONE
   
   TYPE(acc_kind), DIMENSION(:), ALLOCATABLE    :: kinds
   TYPE(acc_device), DIMENSION(:), ALLOCATABLE  :: devices
   INTEGER                                      :: mpi_err
   INTEGER                                      :: world_rank, world_size
   REAL(C_DOUBLE)                               :: cell_mat(3, 3)
   REAL(C_DOUBLE), DIMENSION(:)   , ALLOCATABLE :: exponents
   REAL(C_DOUBLE), DIMENSION(:, :), ALLOCATABLE :: positions, coefficients
   INTEGER                                      :: shell_offsets(0:LIBHERIC_MAX_AM), shell_sizes(0:LIBHERIC_MAX_AM)
   INTEGER                                      :: num_exponents, num_positions, max_shells, max_sets, num_kinds, l_max
   INTEGER                                      :: quartet(4), quartet_permutations(5, 8), num_quartet_permutations
   INTEGER, DIMENSION(:)         , ALLOCATABLE  :: kind_offsets, kind_sizes, atom_kinds
   INTEGER, DIMENSION(:, :)      , ALLOCATABLE  :: set_offsets, set_sizes, set_lmin, set_lmax
   !INTEGER, DIMENSION(:, :, :, :), ALLOCATABLE  :: canonical_idx
   !INTEGER, DIMENSION(:, :)      , ALLOCATABLE  :: canonical_idx_inverse
   !INTEGER,                                     :: n_canonical


   CALL MPI_Init(mpi_err)

   CALL MPI_Comm_size(MPI_COMM_WORLD, world_size, mpi_err)
   CALL MPI_Comm_rank(MPI_COMM_WORLD, world_rank, mpi_err)
   
   WRITE(*, "('Rank ',I3,' in MPI_COMM_WORLD of size ',I3)") world_rank, world_size

   num_exponents = 15
   num_positions = 2
   max_shells    = 6
   max_sets      = 5
   num_kinds     = 2
   l_max         = 3

   ALLOCATE(kind_offsets(num_kinds))
   ALLOCATE(kind_sizes(num_kinds))
   ALLOCATE(set_offsets(max_sets, num_kinds))
   ALLOCATE(set_sizes(max_sets, num_kinds))
   ALLOCATE(set_lmin(max_sets, num_kinds))
   ALLOCATE(set_lmax(max_sets, num_kinds))

   kind_offsets = [ 0, 9 ]
   kind_sizes = [ 5, 4 ]
   set_offsets(:, 1)  = [ 0, 3, 5, 7, 8 ]
   set_offsets(:, 2)  = [ 0, 2, 4, 5, 6 ]

   set_sizes(:, 1) = [ 3, 2, 2, 1, 1 ]   
   set_sizes(:, 2) = [ 2, 2, 1, 1, 0 ] 

   cell_mat = reshape( [ 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 ], shape(cell_mat))
   ALLOCATE(exponents(num_exponents))
   ! kind1
   exponents(1:3) = [ 0.1, 0.2, 0.3 ] ! set1
   exponents(4:5) = [ 0.4, 0.5 ]      ! set2
   exponents(6:7) = [ 0.6, 0.7 ]      ! set3
   exponents(8)  = 0.8  ! set4
   exponents(9)  = 0.9  ! set5

   ! kind2
   exponents(10:11) = [ 1.0, 1.1 ] ! set 1
   exponents(12:13) = [ 1.2, 1.3 ] ! set 2
   exponents(14) = 1.4 ! set 3
   exponents(15) = 1.5 ! set 4

   ALLOCATE(positions(3, num_positions))
   ALLOCATE(atom_kinds(num_positions))
   init_pos: BLOCK
      INTEGER :: x_pos, y_pos, z_pos, ctr = 1
      DO x_pos = 0, CEILING(num_positions ** (1./3.))
         DO y_pos = 0, CEILING(num_positions ** (1./3.))
            DO z_pos = 0, CEILING(num_positions ** (1./3.))
               positions(:, ctr) = [ x_pos + 0.1 , y_pos + 0.1, z_pos + 0.1 ]
               atom_kinds(ctr) = MODULO(ctr, num_kinds)
               IF(ctr .GE. num_positions) EXIT init_pos
               ctr = ctr + 1
            END DO
         END DO
      END DO
      WRITE(*, *) "[ERROR] too few positions were generated"
      CALL EXIT(1)

   END BLOCK init_pos 

   ALLOCATE(coefficients(num_exponents, max_shells))
   coefficients = 0

   !kind1 - set1: 2s, 1p
   coefficients(1:3, 1) = [ 1.6, 1.7, 1.8 ]
   coefficients(1:3, 2) = [ 1.9, 2.0, 2.1 ]
   coefficients(1:3, 3) = [ 2.1, 2.3, 2.4 ]

   !kind1 - set2: 1s, 1p
   coefficients(4:5, 1) = [ 2.5, 2.6 ]
   coefficients(4:5, 3) = [ 2.7, 2.8 ]

   !kind1 - set3: 1p, 1d
   coefficients(6:7, 3) = [ 2.9, 3.0 ]
   coefficients(6:7, 5) = [ 3.1, 3.2 ]
   
   !kind1 - set4: 1d
   coefficients(8, 5) = 3.3
   
   !kind1 - set5: 1f
   coefficients(9, 6) = 3.4
   
   !kind2 - set1: 1s, 1p
   coefficients(10:11, 1) = [ 3.5, 3.6 ]
   coefficients(10:11, 3) = [ 3.7, 3.8 ]
   
   !kind2 - set2: 2p
   coefficients(12:13, 3) = [ 3.9, 4.0 ]
   coefficients(12:13, 4) = [ 4.1, 4.2 ]

   !kind2 - set3: 1d
   coefficients(14, 5) = 4.3
   
   !kind2 - set4: 1f
   coefficients(15, 6) = 4.4

   set_lmin(:, 1) = [ 0, 0, 1, 2, 3 ]
   set_lmax(:, 1) = [ 1, 1, 2, 2, 3 ]
   set_lmin(:, 2) = [ 0, 1, 2, 3, -1 ]
   set_lmax(:, 2) = [ 1, 1, 2, 3, -1 ]

   shell_offsets = [ 0, 2, 4, 5 ]
   shell_sizes   = [ 2, 2, 1, 1 ]

   ALLOCATE(kinds(5)) 
   kinds(1) = acc_kind("A100", "CUDA", "libintx")
   kinds(2) = acc_kind("MI210","HIP","libintx")
   kinds(3) = acc_kind("P520","ONEAPI","intel_eris")
   kinds(4) = acc_kind("U280","XRT","seri")
   kinds(5) = acc_kind("VCK5000","XRT","eri_aie")
    
   IF (world_rank == 0) CALL pc2_hide_acl()

    
   BLOCK
      INTEGER :: i, r, ctx_functional
      INTEGER :: num_contexts = 1
      INTEGER :: num_neighbor_cells
      TYPE(C_PTR), DIMENSION(:), ALLOCATABLE :: ctx_handles
      ALLOCATE(ctx_handles(num_contexts))
      DO r = world_size - 1, 0, -1
         IF (world_rank .EQ. r) THEN
            CALL query_accelerator_devices(kinds, devices)
            WRITE(*, "('Discovered ',I2,' devices : ')") SIZE(devices)
            DO i = 1, SIZE(devices)
               WRITE(*, "(' -> DEVICE @ ',A12,' of kind (',A10,', ',A10,', ',A10,') -- ',A,' on ',I3,'x ',A)") &
                  TRIM(devices(i)%bdf), &
                  TRIM(devices(i)%kind%name), &
                  TRIM(devices(i)%kind%type), &
                  TRIM(devices(i)%kind%library), &
                  TRIM(devices(i)%name), &
                  devices(i)%num_cores, &
                  TRIM(devices(i)%cpu_name)
               IF (( &
                      TRIM(devices(i)%kind%name) .EQ. "P520" .AND. &
                      TRIM(devices(i)%bdf) .EQ. "0000:00:00.0" .AND. &
                      world_rank == 0 &
                   ) &
                   .OR. &
                   ( &
                      TRIM(devices(i)%kind%name) .EQ. "P520" .AND. &
                      TRIM(devices(i)%bdf) .EQ. "0000:00:01.0" .AND. &
                      world_rank == 1 &
                   )) THEN
                  CALL create_accelerator_contexts(ctx_handles(1), devices(i), &
                                                   num_contexts, &
                                                   cell_mat, &
                                                   positions, atom_kinds, &
                                                   exponents, &
                                                   coefficients, &
                                                   kind_offsets, kind_sizes, &
                                                   set_offsets, set_sizes, &
                                                   shell_offsets, shell_sizes)
               END IF
            END DO
         END IF
         CALL MPI_Barrier(MPI_COMM_WORLD, mpi_err)
      END DO
      quartet = [ 2, 2, 2, 1 ]

      !CALL generate_canonical_ids(l_max, canonical_idx, canonical_idx_inverse, n_canonical)
      CALL generate_quartet_permutations(quartet, quartet_permutations, num_quartet_permutations)

      !!$OMP PARALLEL default(shared)
      !WRITE(*, "('Hello from thread #',I2)") OMP_get_thread_num()
      !IF (OMP_get_thread_num() .LT. num_contexts) THEN 
      !gen_quartets: BLOCK
      !   TYPE(C_PTR), VALUE :: ctx = ctx_handles(OMP_get_thread_num() + 1)
      !   INTEGER :: la, lb, lc, ld 
      !   INTEGER :: perm_la, perm_lb, perm_lc, perm_ld 
      !   INTEGER :: A, B, C, D 
      !   INTEGER :: kind_A, kind_B, kind_C, kind_D 
      !   INTEGER :: kind_A_off, kind_B_off, kind_C_off, kind_D_off 
      !   INTEGER :: kind_A_size, kind_B_size, kind_C_size, kind_D_size
      !   INTEGER :: set_A_ctr, set_B_ctr, set_C_ctr, set_D_ctr 
      !   INTEGER :: set_A_size, set_B_size, set_C_size, set_D_size 
      !   INTEGER :: exp_A_off, exp_B_off, exp_C_off, exp_D_off 
      !   INTEGER :: shell_A_ctr, shell_B_ctr, shell_C_ctr, shell_D_ctr 
      !   INTEGER :: shell_A_off, shell_B_off, shell_C_off, shell_D_off 
      !   INTEGER :: exp_A, exp_B, exp_C, exp_D 
      !   INTEGER :: B_pos_off_x, B_pos_off_y, B_pos_off_z
      !   INTEGER :: C_pos_off_x, C_pos_off_y, C_pos_off_z
      !   INTEGER :: D_pos_off_x, D_pos_off_y, D_pos_off_z
      !   INTEGER :: perm, perm_idx
      !   REAL(C_DOUBLE), PARAMETER :: eps = 10e-12
      !   la = quartet(1)
      !   lb = quartet(2)
      !   lc = quartet(3)
      !   ld = quartet(4)
      !   CALL accelerator_context_set_quartet(ctx, la, lb, lc, ld)
      !   CALL accelerator_context_functional(ctx, ctx_functional)
      !   IF (ctx_functional .EQ. 0) THEN 
      !      WRITE (*, "('[ERROR] Failed to set quartet (',I1,I1,'|',I1,I1') for permutation = ',I1)") la, lb, lc, ld, 0
      !      CALL EXIT(1)
      !   END IF
      !   DO perm_idx = 1, num_quartet_permutations
      !      perm_la = quartet_permutations(1, perm_idx)
      !      perm_lb = quartet_permutations(2, perm_idx)
      !      perm_lc = quartet_permutations(3, perm_idx)
      !      perm_ld = quartet_permutations(4, perm_idx)
      !      perm = quartet_permutations(5, perm_idx)
      !      CALL accelerator_context_set_quartet_permutation(ctx, perm) 
      !      CALL accelerator_context_functional(ctx, ctx_functional)
      !      IF (ctx_functional .EQ. 0) THEN 
      !         WRITE (*, "('[ERROR] Failed to set quartet (',I1,I1,'|',I1,I1') for permutation = ',I1)") la, lb, lc, ld, perm
      !         CYCLE
      !      END IF
      !      DO A = 1, SIZE(positions, 2)
      !      CALL accelerator_context_set_atom_id(ctx, 0, A-1)
      !      DO B = A, SIZE(positions, 2)
      !      CALL accelerator_context_set_atom_id(ctx, 1, B-1)
      !      DO C = 1, SIZE(positions, 2)
      !      CALL accelerator_context_set_atom_id(ctx, 2, C-1)
      !      DO D = C, SIZE(positions, 2)
      !      CALL accelerator_context_set_atom_id(ctx, 3, D-1)
      !         IF (A + B .GT. C + D) CYCLE
      !         IF (A + B .EQ. C + D .AND. A .GT. C) CYCLE

      !         !WRITE(*, "('ABCD = ',I1,',',I1,',',I1,',',I1)") A, B, C, D

      !         kind_A = atom_kinds(A) + 1
      !         kind_B = atom_kinds(B) + 1
      !         kind_C = atom_kinds(C) + 1
      !         kind_D = atom_kinds(D) + 1
      !         
      !         !WRITE(*, "('kinds = ',I1,',',I1,',',I1,',',I1)") kind_A, kind_B, kind_C, kind_D

      !         kind_A_off = kind_offsets(kind_A) + 1
      !         kind_B_off = kind_offsets(kind_B) + 1
      !         kind_C_off = kind_offsets(kind_C) + 1
      !         kind_D_off = kind_offsets(kind_D) + 1 
      !         
      !         !WRITE(*, "('kind_offs = ',I2,',',I2,',',I2,',',I2)") kind_A_off, kind_B_off, kind_C_off, kind_D_off

      !         kind_A_size = kind_sizes(kind_A)
      !         kind_B_size = kind_sizes(kind_B)
      !         kind_C_size = kind_sizes(kind_C)
      !         kind_D_size = kind_sizes(kind_D)
      !         
      !         !WRITE(*, "('kind_sizes = ',I1,',',I1,',',I1,',',I1)") kind_A_size, kind_B_size, kind_C_size, kind_D_size

      !         DO set_A_ctr = 1, kind_A_size
      !         IF (perm_la .LT. set_lmin(set_A_ctr, kind_A) .OR. perm_la .GT. set_lmax(set_A_ctr, kind_A)) CYCLE
      !         set_A_size = set_sizes(set_A_ctr, kind_A)
      !         exp_a_off = kind_A_off + set_offsets(set_A_ctr, kind_A)
      !         CALL accelerator_context_set_set_id(ctx, 0, set_A_ctr - 1) 
      !         
      !         DO set_B_ctr = 1, kind_B_size
      !         IF (perm_lb .LT. set_lmin(set_B_ctr, kind_B) .OR. perm_lb .GT. set_lmax(set_B_ctr, kind_B)) CYCLE
      !         set_B_size = set_sizes(set_B_ctr, kind_B)
      !         exp_b_off = kind_B_off + set_offsets(set_B_ctr, kind_B)
      !         CALL accelerator_context_set_set_id(ctx, 1, set_B_ctr - 1) 
      !         
      !         DO set_C_ctr = 1, kind_C_size
      !         IF (perm_lc .LT. set_lmin(set_C_ctr, kind_C) .OR. perm_lc .GT. set_lmax(set_C_ctr, kind_C)) CYCLE
      !         set_C_size = set_sizes(set_C_ctr, kind_C)
      !         exp_c_off = kind_C_off + set_offsets(set_C_ctr, kind_C)
      !         CALL accelerator_context_set_set_id(ctx, 2, set_C_ctr - 1) 
      !         
      !         DO set_D_ctr = 1, kind_D_size
      !         IF (perm_ld .LT. set_lmin(set_D_ctr, kind_D) .OR. perm_ld .GT. set_lmax(set_D_ctr, kind_D)) CYCLE
      !         set_D_size = set_sizes(set_D_ctr, kind_D)
      !         exp_d_off = kind_D_off + set_offsets(set_D_ctr, kind_D)
      !         CALL accelerator_context_set_set_id(ctx, 3, set_D_ctr - 1) 
      !            
      !            !WRITE(*, "('set_ctrs  = ',I1,',',I1,',',I1,',',I1)") set_A_ctr, set_B_ctr, set_C_ctr, set_D_ctr
      !            !WRITE(*, "('set_sizes = ',I1,',',I1,',',I1,',',I1)") set_A_size, set_B_size, set_C_size, set_D_size
      !         
      !            DO shell_a_ctr = 1, shell_sizes(la)
      !            shell_a_off = shell_offsets(la) + shell_a_ctr
      !            IF (ALL(ABS(coefficients(exp_a_off:exp_a_off + set_A_size - 1, shell_a_off)) .LT. eps)) CYCLE
      !            CALL accelerator_context_set_shell_id(ctx, 0, shell_a_ctr - 1)
      !            
      !            DO shell_b_ctr = 1, shell_sizes(lb)
      !            shell_b_off = shell_offsets(lb) + shell_b_ctr
      !            IF (ALL(ABS(coefficients(exp_b_off:exp_b_off + set_B_size - 1, shell_b_off)) .LT. eps)) CYCLE
      !            CALL accelerator_context_set_shell_id(ctx, 1, shell_b_ctr - 1)
      !            
      !            DO shell_c_ctr = 1, shell_sizes(lc)
      !            shell_c_off = shell_offsets(lc) + shell_c_ctr
      !            IF (ALL(ABS(coefficients(exp_c_off:exp_c_off + set_C_size - 1, shell_c_off)) .LT. eps)) CYCLE
      !            CALL accelerator_context_set_shell_id(ctx, 2, shell_c_ctr - 1)
      !            
      !            DO shell_d_ctr = 1, shell_sizes(ld)
      !            shell_d_off = shell_offsets(ld) + shell_d_ctr
      !            IF (ALL(ABS(coefficients(exp_d_off:exp_d_off + set_D_size - 1, shell_d_off)) .LT. eps)) CYCLE
      !            CALL accelerator_context_set_shell_id(ctx, 3, shell_d_ctr - 1)
      !            
      !               DO exp_a = 0, set_A_size - 1
      !               IF (ABS(coefficients(exp_a_off + exp_a, shell_a_off)) .LT. eps) CYCLE
      !               CALL accelerator_context_set_exponent_id(ctx, 0, exp_a) 
      !               
      !               DO exp_b = 0, set_B_size - 1
      !               IF (ABS(coefficients(exp_b_off + exp_b, shell_b_off)) .LT. eps) CYCLE
      !               CALL accelerator_context_set_exponent_id(ctx, 1, exp_b) 
      !               
      !               DO exp_c = 0, set_C_size - 1
      !               IF (ABS(coefficients(exp_c_off + exp_c, shell_c_off)) .LT. eps) CYCLE
      !               CALL accelerator_context_set_exponent_id(ctx, 2, exp_c) 
      !               
      !               DO exp_d = 0, set_D_size - 1
      !               IF (ABS(coefficients(exp_d_off + exp_d, shell_d_off)) .LT. eps) CYCLE
      !               CALL accelerator_context_set_exponent_id(ctx, 3, exp_d) 
      !                  DO B_pos_off_x = 0, 1
      !                  DO B_pos_off_y = 0, 0
      !                  DO B_pos_off_z = 0, 0
      !                     CALL accelerator_context_set_offset(ctx, 0, B_pos_off_x, B_pos_off_y, B_pos_off_z) 
      !                  DO C_pos_off_x = 0, 0
      !                  DO C_pos_off_y = 0, 0
      !                  DO C_pos_off_z = 0, 0 
      !                     CALL accelerator_context_set_offset(ctx, 1, C_pos_off_x, C_pos_off_y, C_pos_off_z) 
      !                  DO D_pos_off_x = 0, 0
      !                  DO D_pos_off_y = 0, 0
      !                  DO D_pos_off_z = 0, 1
      !                     CALL accelerator_context_set_offset(ctx, 2, D_pos_off_x, D_pos_off_y, D_pos_off_z)
      !                     CALL accelerator_context_submit_peri(ctx)
      !                  END DO
      !                  END DO
      !                  END DO
      !                  END DO
      !                  END DO
      !                  END DO
      !                  END DO
      !                  END DO
      !                  END DO
      !               END DO
      !               END DO
      !               END DO
      !               END DO
      !            END DO
      !            END DO
      !            END DO 
      !            END DO
      !         END DO
      !         END DO
      !         END DO
      !         END DO
      !      END DO
      !      END DO
      !      END DO
      !      END DO
      !      CALL accelerator_context_finish_quartet_class(ctx)
      !      EXIT gen_quartets
      !   END DO 
      !END BLOCK gen_quartets
      !ELSE 
      !consume_quartets: BLOCK
      !   TYPE(C_PTR), VALUE :: ctx = ctx_handles(OMP_get_thread_num() - (num_contexts - 1) + 1)
      !   INTEGER :: la, lb, lc, ld
      !   INTEGER :: nsa, nsb, nsc, nsd 
      !   INTEGER :: perm_la, perm_lb, perm_lc, perm_ld 
      !   INTEGER :: A, B, C, D 
      !   INTEGER :: kind_A, kind_B, kind_C, kind_D 
      !   INTEGER :: kind_A_off, kind_B_off, kind_C_off, kind_D_off 
      !   INTEGER :: kind_A_size, kind_B_size, kind_C_size, kind_D_size
      !   INTEGER :: set_A_ctr, set_B_ctr, set_C_ctr, set_D_ctr 
      !   INTEGER :: set_A_size, set_B_size, set_C_size, set_D_size 
      !   INTEGER :: exp_A_off, exp_B_off, exp_C_off, exp_D_off 
      !   INTEGER :: shell_A_ctr, shell_B_ctr, shell_C_ctr, shell_D_ctr 
      !   INTEGER :: shell_A_off, shell_B_off, shell_C_off, shell_D_off 
      !   INTEGER :: perm, perm_idx
      !   TYPE(C_PTR) :: c_integral_buffer
      !   REAL(C_DOUBLE), DIMENSION(:), POINTER :: integral_buffer
      !   REAL(C_DOUBLE), PARAMETER :: eps = 10e-12
      !   la = quartet(1)
      !   lb = quartet(2)
      !   lc = quartet(3)
      !   ld = quartet(4)

      !   nsa = LIBHERIC_NSO(la + 1)
      !   nsb = LIBHERIC_NSO(lb + 1)
      !   nsc = LIBHERIC_NSO(lc + 1)
      !   nsd = LIBHERIC_NSO(ld + 1)

      !   DO perm_idx = 1, num_quartet_permutations
      !      perm_la = quartet_permutations(1, perm_idx)
      !      perm_lb = quartet_permutations(2, perm_idx)
      !      perm_lc = quartet_permutations(3, perm_idx)
      !      perm_ld = quartet_permutations(4, perm_idx)
      !      perm = quartet_permutations(5, perm_idx)
      !      ctx_functional = 0
      !      DO WHILE (ctx_functional == 0) 
      !         CALL accelerator_context_functional(ctx, ctx_functional)
      !      END DO 
      !      IF (ctx_functional .EQ. 0) THEN 
      !         WRITE (*, "('[ERROR] Failed to set quartet (',I1,I1,'|',I1,I1') for permutation = ',I1)") la, lb, lc, ld, perm
      !         CYCLE
      !      END IF
      !      DO A = 1, SIZE(positions, 2)
      !      DO B = A, SIZE(positions, 2)
      !      DO C = 1, SIZE(positions, 2)
      !      DO D = C, SIZE(positions, 2)
      !         IF (A + B .GT. C + D) CYCLE
      !         IF (A + B .EQ. C + D .AND. A .GT. C) CYCLE
      !         kind_A = atom_kinds(A) + 1
      !         kind_B = atom_kinds(B) + 1
      !         kind_C = atom_kinds(C) + 1
      !         kind_D = atom_kinds(D) + 1
      !         
      !         kind_A_off = kind_offsets(kind_A) + 1
      !         kind_B_off = kind_offsets(kind_B) + 1
      !         kind_C_off = kind_offsets(kind_C) + 1
      !         kind_D_off = kind_offsets(kind_D) + 1 
      !         
      !         kind_A_size = kind_sizes(kind_A)
      !         kind_B_size = kind_sizes(kind_B)
      !         kind_C_size = kind_sizes(kind_C)
      !         kind_D_size = kind_sizes(kind_D)
      !         
      !         CALL accelerator_context_get_quartet(ctx, A, B, C, D, c_integral_buffer)
      !         
      !         
      !         !DO set_A_ctr = 1, kind_A_size
      !         !IF (perm_la .LT. set_lmin(set_A_ctr, kind_A) .OR. perm_la .GT. set_lmax(set_A_ctr, kind_A)) CYCLE
      !         !set_A_size = set_sizes(set_A_ctr, kind_A)
      !         !exp_a_off = kind_A_off + set_offsets(set_A_ctr, kind_A)
      !         !
      !         !DO set_B_ctr = 1, kind_B_size
      !         !IF (perm_lb .LT. set_lmin(set_B_ctr, kind_B) .OR. perm_lb .GT. set_lmax(set_B_ctr, kind_B)) CYCLE
      !         !set_B_size = set_sizes(set_B_ctr, kind_B)
      !         !exp_b_off = kind_B_off + set_offsets(set_B_ctr, kind_B)
      !         !
      !         !DO set_C_ctr = 1, kind_C_size
      !         !IF (perm_lc .LT. set_lmin(set_C_ctr, kind_C) .OR. perm_lc .GT. set_lmax(set_C_ctr, kind_C)) CYCLE
      !         !set_C_size = set_sizes(set_C_ctr, kind_C)
      !         !exp_c_off = kind_C_off + set_offsets(set_C_ctr, kind_C)
      !         !
      !         !DO set_D_ctr = 1, kind_D_size
      !         !IF (perm_ld .LT. set_lmin(set_D_ctr, kind_D) .OR. perm_ld .GT. set_lmax(set_D_ctr, kind_D)) CYCLE
      !         !set_D_size = set_sizes(set_D_ctr, kind_D)
      !         !exp_d_off = kind_D_off + set_offsets(set_D_ctr, kind_D)
      !         !   
      !         !   DO shell_a_ctr = 1, shell_sizes(la)
      !         !   shell_a_off = shell_offsets(la) + shell_a_ctr
      !         !   IF (ALL(ABS(coefficients(exp_a_off:exp_a_off + set_A_size - 1, shell_a_off)) .LT. eps)) CYCLE
      !         !   
      !         !   DO shell_b_ctr = 1, shell_sizes(lb)
      !         !   shell_b_off = shell_offsets(lb) + shell_b_ctr
      !         !   IF (ALL(ABS(coefficients(exp_b_off:exp_b_off + set_B_size - 1, shell_b_off)) .LT. eps)) CYCLE
      !         !   
      !         !   DO shell_c_ctr = 1, shell_sizes(lc)
      !         !   shell_c_off = shell_offsets(lc) + shell_c_ctr
      !         !   IF (ALL(ABS(coefficients(exp_c_off:exp_c_off + set_C_size - 1, shell_c_off)) .LT. eps)) CYCLE
      !         !   
      !         !   DO shell_d_ctr = 1, shell_sizes(ld)
      !         !   shell_d_off = shell_offsets(ld) + shell_d_ctr
      !         !   IF (ALL(ABS(coefficients(exp_d_off:exp_d_off + set_D_size - 1, shell_d_off)) .LT. eps)) CYCLE
      !         !   
      !         !      CALL accelerator_context_get_next_ceri(ctx, integral_buffer(1))
      !         !   END DO
      !         !   END DO
      !         !   END DO 
      !         !   END DO
      !         !END DO
      !         !END DO
      !         !END DO
      !         !END DO
      !      END DO
      !      END DO
      !      END DO
      !      END DO
      !      EXIT consume_quartets
      !   END DO 
      !END BLOCK consume_quartets
      !END IF
      !!$OMP END PARALLEL
      
      !$OMP PARALLEL default(shared)
      WRITE(*, "('Hello from thread #',I2)") OMP_get_thread_num()
      process_quartets: BLOCK
         TYPE(C_PTR), VALUE :: ctx = ctx_handles(OMP_get_thread_num() + 1)
         INTEGER :: la, lb, lc, ld 
         INTEGER :: perm_la, perm_lb, perm_lc, perm_ld 
         INTEGER :: A, B, C, D 
         INTEGER :: kind_A, kind_B, kind_C, kind_D 
         INTEGER :: kind_A_off, kind_B_off, kind_C_off, kind_D_off 
         INTEGER :: kind_A_size, kind_B_size, kind_C_size, kind_D_size
         INTEGER :: set_A_ctr, set_B_ctr, set_C_ctr, set_D_ctr 
         INTEGER :: set_A_size, set_B_size, set_C_size, set_D_size 
         INTEGER :: exp_A_off, exp_B_off, exp_C_off, exp_D_off 
         INTEGER :: shell_A_ctr, shell_B_ctr, shell_C_ctr, shell_D_ctr 
         INTEGER :: shell_A_off, shell_B_off, shell_C_off, shell_D_off 
         INTEGER :: exp_A, exp_B, exp_C, exp_D 
         INTEGER :: B_pos_off_x, B_pos_off_y, B_pos_off_z
         INTEGER :: C_pos_off_x, C_pos_off_y, C_pos_off_z
         INTEGER :: D_pos_off_x, D_pos_off_y, D_pos_off_z
         INTEGER :: perm, perm_idx
         REAL(C_DOUBLE), PARAMETER :: eps = 10e-12
         la = quartet(1)
         lb = quartet(2)
         lc = quartet(3)
         ld = quartet(4)
         CALL accelerator_context_set_quartet(ctx, la, lb, lc, ld)
         CALL accelerator_context_functional(ctx, ctx_functional)
         IF (ctx_functional .EQ. 0) THEN 
            WRITE (*, "('[ERROR] Failed to set quartet (',I1,I1,'|',I1,I1') for permutation = ',I1)") la, lb, lc, ld, 0
            CALL EXIT(1)
         END IF
         DO perm_idx = 1, num_quartet_permutations
            perm_la = quartet_permutations(1, perm_idx)
            perm_lb = quartet_permutations(2, perm_idx)
            perm_lc = quartet_permutations(3, perm_idx)
            perm_ld = quartet_permutations(4, perm_idx)
            perm = quartet_permutations(5, perm_idx)
            CALL accelerator_context_set_quartet_permutation(ctx, perm) 
            CALL accelerator_context_functional(ctx, ctx_functional)
            IF (ctx_functional .EQ. 0) THEN 
               WRITE (*, "('[ERROR] Failed to set quartet (',I1,I1,'|',I1,I1') for permutation = ',I1)") la, lb, lc, ld, perm
               CYCLE
            END IF
            DO A = 1, SIZE(positions, 2)
            CALL accelerator_context_set_atom_id(ctx, 0, A-1)
            DO B = A, SIZE(positions, 2)
            CALL accelerator_context_set_atom_id(ctx, 1, B-1)
            DO C = 1, SIZE(positions, 2)
            CALL accelerator_context_set_atom_id(ctx, 2, C-1)
            DO D = C, SIZE(positions, 2)
            CALL accelerator_context_set_atom_id(ctx, 3, D-1)
               IF (A + B .GT. C + D) CYCLE
               IF (A + B .EQ. C + D .AND. A .GT. C) CYCLE

               !WRITE(*, "('ABCD = ',I1,',',I1,',',I1,',',I1)") A, B, C, D

               kind_A = atom_kinds(A) + 1
               kind_B = atom_kinds(B) + 1
               kind_C = atom_kinds(C) + 1
               kind_D = atom_kinds(D) + 1
               
               !WRITE(*, "('kinds = ',I1,',',I1,',',I1,',',I1)") kind_A, kind_B, kind_C, kind_D

               kind_A_off = kind_offsets(kind_A) + 1
               kind_B_off = kind_offsets(kind_B) + 1
               kind_C_off = kind_offsets(kind_C) + 1
               kind_D_off = kind_offsets(kind_D) + 1 
               
               !WRITE(*, "('kind_offs = ',I2,',',I2,',',I2,',',I2)") kind_A_off, kind_B_off, kind_C_off, kind_D_off

               kind_A_size = kind_sizes(kind_A)
               kind_B_size = kind_sizes(kind_B)
               kind_C_size = kind_sizes(kind_C)
               kind_D_size = kind_sizes(kind_D)
               
               !WRITE(*, "('kind_sizes = ',I1,',',I1,',',I1,',',I1)") kind_A_size, kind_B_size, kind_C_size, kind_D_size

               DO set_A_ctr = 1, kind_A_size
               IF (perm_la .LT. set_lmin(set_A_ctr, kind_A) .OR. perm_la .GT. set_lmax(set_A_ctr, kind_A)) CYCLE
               set_A_size = set_sizes(set_A_ctr, kind_A)
               exp_a_off = kind_A_off + set_offsets(set_A_ctr, kind_A)
               CALL accelerator_context_set_set_id(ctx, 0, set_A_ctr - 1) 
               
               DO set_B_ctr = 1, kind_B_size
               IF (perm_lb .LT. set_lmin(set_B_ctr, kind_B) .OR. perm_lb .GT. set_lmax(set_B_ctr, kind_B)) CYCLE
               set_B_size = set_sizes(set_B_ctr, kind_B)
               exp_b_off = kind_B_off + set_offsets(set_B_ctr, kind_B)
               CALL accelerator_context_set_set_id(ctx, 1, set_B_ctr - 1) 
               
               DO set_C_ctr = 1, kind_C_size
               IF (perm_lc .LT. set_lmin(set_C_ctr, kind_C) .OR. perm_lc .GT. set_lmax(set_C_ctr, kind_C)) CYCLE
               set_C_size = set_sizes(set_C_ctr, kind_C)
               exp_c_off = kind_C_off + set_offsets(set_C_ctr, kind_C)
               CALL accelerator_context_set_set_id(ctx, 2, set_C_ctr - 1) 
               
               DO set_D_ctr = 1, kind_D_size
               IF (perm_ld .LT. set_lmin(set_D_ctr, kind_D) .OR. perm_ld .GT. set_lmax(set_D_ctr, kind_D)) CYCLE
               set_D_size = set_sizes(set_D_ctr, kind_D)
               exp_d_off = kind_D_off + set_offsets(set_D_ctr, kind_D)
               CALL accelerator_context_set_set_id(ctx, 3, set_D_ctr - 1) 
                  
                  !WRITE(*, "('set_ctrs  = ',I1,',',I1,',',I1,',',I1)") set_A_ctr, set_B_ctr, set_C_ctr, set_D_ctr
                  !WRITE(*, "('set_sizes = ',I1,',',I1,',',I1,',',I1)") set_A_size, set_B_size, set_C_size, set_D_size
                  DO 
                     accerelerator_context_ready_for_new_input(ctx, can_produce)
                     IF (can_produce) THEN
                        DO shell_a_ctr = 1, shell_sizes(la)
                        shell_a_off = shell_offsets(la) + shell_a_ctr
                        IF (ALL(ABS(coefficients(exp_a_off:exp_a_off + set_A_size - 1, shell_a_off)) .LT. eps)) CYCLE
                        CALL accelerator_context_set_shell_id(ctx, 0, shell_a_ctr - 1)
                        
                        DO shell_b_ctr = 1, shell_sizes(lb)
                        shell_b_off = shell_offsets(lb) + shell_b_ctr
                        IF (ALL(ABS(coefficients(exp_b_off:exp_b_off + set_B_size - 1, shell_b_off)) .LT. eps)) CYCLE
                        CALL accelerator_context_set_shell_id(ctx, 1, shell_b_ctr - 1)
                        
                        DO shell_c_ctr = 1, shell_sizes(lc)
                        shell_c_off = shell_offsets(lc) + shell_c_ctr
                        IF (ALL(ABS(coefficients(exp_c_off:exp_c_off + set_C_size - 1, shell_c_off)) .LT. eps)) CYCLE
                        CALL accelerator_context_set_shell_id(ctx, 2, shell_c_ctr - 1)
                        
                        DO shell_d_ctr = 1, shell_sizes(ld)
                        shell_d_off = shell_offsets(ld) + shell_d_ctr
                        IF (ALL(ABS(coefficients(exp_d_off:exp_d_off + set_D_size - 1, shell_d_off)) .LT. eps)) CYCLE
                        CALL accelerator_context_set_shell_id(ctx, 3, shell_d_ctr - 1)
                        
                           DO exp_a = 0, set_A_size - 1
                           IF (ABS(coefficients(exp_a_off + exp_a, shell_a_off)) .LT. eps) CYCLE
                           CALL accelerator_context_set_exponent_id(ctx, 0, exp_a) 
                           
                           DO exp_b = 0, set_B_size - 1
                           IF (ABS(coefficients(exp_b_off + exp_b, shell_b_off)) .LT. eps) CYCLE
                           CALL accelerator_context_set_exponent_id(ctx, 1, exp_b) 
                           
                           DO exp_c = 0, set_C_size - 1
                           IF (ABS(coefficients(exp_c_off + exp_c, shell_c_off)) .LT. eps) CYCLE
                           CALL accelerator_context_set_exponent_id(ctx, 2, exp_c) 
                           
                           DO exp_d = 0, set_D_size - 1
                           IF (ABS(coefficients(exp_d_off + exp_d, shell_d_off)) .LT. eps) CYCLE
                           CALL accelerator_context_set_exponent_id(ctx, 3, exp_d) 
                              DO B_pos_off_x = 0, 1
                              DO B_pos_off_y = 0, 0
                              DO B_pos_off_z = 0, 0
                                 CALL accelerator_context_set_offset(ctx, 0, B_pos_off_x, B_pos_off_y, B_pos_off_z) 
                              DO C_pos_off_x = 0, 0
                              DO C_pos_off_y = 0, 0
                              DO C_pos_off_z = 0, 0 
                                 CALL accelerator_context_set_offset(ctx, 1, C_pos_off_x, C_pos_off_y, C_pos_off_z) 
                              DO D_pos_off_x = 0, 0
                              DO D_pos_off_y = 0, 0
                              DO D_pos_off_z = 0, 1
                                 CALL accelerator_context_set_offset(ctx, 2, D_pos_off_x, D_pos_off_y, D_pos_off_z)
                                 CALL accelerator_context_submit_peri(ctx)
                              END DO
                              END DO
                              END DO
                              END DO
                              END DO
                              END DO
                              END DO
                              END DO
                              END DO
                           END DO
                           END DO
                           END DO
                           END DO
                        END DO
                        END DO
                        END DO 
                        END DO
                     END IF
                     accelerator_context_compute()
                     IF (can_produce) EXIT
                  END DO
               END DO
               END DO
               END DO
               END DO
               accelerator_context_consume()
            END DO
            END DO
            END DO
            END DO
            CALL accelerator_context_finish_quartet_class(ctx)
            EXIT gen_quartets
         END DO 
      END BLOCK process_quartets
      !$OMP END PARALLEL
        

      IF(ctx_functional .NE. 0) THEN
         WRITE(*, "('Rank #',I1,' set quartet and is functional')") world_rank
      ELSE
         WRITE(*, "('Rank #',I1,' failed to set quartet and is not functional!')") world_rank
      ENDIF 
      DO r = world_size - 1, 0, -1
         IF (world_rank .EQ. r) THEN
            CALL free_accelerator_context(ctx)
         END IF
         CALL MPI_Barrier(MPI_COMM_WORLD, mpi_err)
      END DO 
   END BLOCK

!   ALLOCATE(devices(2))

!   devices(1)%name = "BLA"
!   devices(1)%bdf  = "0000:00:00.0"
!   devices(1)%cpu_name = "BLABLA"
!   devices(1)%num_cores = 1
!   devices(1)%kind = kinds(3)
!   
!   devices(2)%name = "BLA2"
!   devices(2)%bdf  = "0000:00:01.0"
!   devices(2)%cpu_name = "BLABLA2"
!   devices(2)%num_cores = 1
!   devices(2)%kind = kinds(3)


   CALL MPI_Finalize(mpi_err)

END PROGRAM
