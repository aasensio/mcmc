module maths
implicit none
contains

!-------------------------------------------------------------
! Initialize the random number generator
!-------------------------------------------------------------
	subroutine init_random_seed()
	integer :: i, n, clock
   integer, dimension(:), allocatable :: seed
          
   	call random_seed(size = n)
      allocate(seed(n))
          
      call system_clock(count=clock)
          
      seed = clock + 37 * (/ (i - 1, i = 1, n) /)
      call random_seed(put = seed)
          
      deallocate(seed)
   end subroutine init_random_seed
          
!-------------------------------------------------------------
! Generates a random number following an uniform distribution in the interval [0,1]
! Call it with idum<0 for initialization
!-------------------------------------------------------------
	function randomu()
	real(kind=8) :: randomu
	
		call random_number(randomu)
							
	end function randomu
	
!-------------------------------------------------------------
! Generates a random number following an normal distribution with zero mean
! and unit variance
!-------------------------------------------------------------
	function randomn()

	real(kind=8) :: randomn

	real(kind=8) :: u, sum
	real(kind=8), save :: v, sln
	logical, save :: second = .false.
	real(kind=8), parameter :: one = 1.0, vsmall = tiny( one )

! if second, use the second random number generated on last call
	if (second) then

		second = .false.
  		randomn = v*sln
	else
! first call; generate a pair of random normals

  		second = .true.
  		do
    		call random_number( u )
    		call random_number( v )
    		u = scale( u, 1 ) - one
    		v = scale( v, 1 ) - one
    		sum = u*u + v*v + vsmall         ! vsmall added to prevent log(zero) / zero
    		if(sum < one) exit
  		end do
  		sln = sqrt(- scale( log(sum), 1 ) / sum)
  		randomn = u*sln
	end if

	return
	end function randomn

!-------------------------------------------------------------
! Carry out the Cholesky decomposition of a symmetric matrix
!-------------------------------------------------------------
	subroutine cholesky(a,n,p)
	integer :: n
	real(kind=8) :: a(n,n), p(n)
	integer :: i, j, k
	real(kind=8) :: sum
	
		do i = 1, n
			do j = i, n
				sum = a(i,j)
				do k = i-1, 1, -1
					sum = sum - a(i,k)*a(j,k)
				enddo
				if (i == j) then
					if (sum == 0.d0) then
						print *, 'Cholesky decomposition failed...'												
					endif
					p(i) = dsqrt(sum)
				else
					a(j,i) = sum / p(i)
				endif
			enddo
		enddo
						
	end subroutine cholesky

!-------------------------------------------------------------
! Generates a multivariate normal random number with a given
! mean and covariance matrix
!-------------------------------------------------------------
	function mrandomn(idum,rmean,covar)
	integer :: idum, n, i, j
	real(kind=8) :: rmean(:), covar(:,:), mrandomn(size(rmean))
	real(kind=8) :: chol(size(rmean),size(rmean)), p(size(rmean)), eps(size(rmean))
	
		n = size(rmean)
				
		chol = covar
		
		do i = 1, n
			eps(i) = randomn()
			chol(i,i) = chol(i,i) + 1.d-7    ! Some regularization
		enddo
								
		call cholesky(chol,n,p)
										
		do j = 1, n			
			do i = j, n
				chol(j,i) = 0.d0
			enddo
			chol(j,j) = p(j)
		enddo
								
		mrandomn = matmul(chol,eps) + rmean			
				
		
	end function mrandomn

end module maths