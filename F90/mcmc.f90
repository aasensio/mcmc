program mcmc_test
use mcmc_class_hierarchical
use maths, only : randomn
implicit none

	type(mcmc_hierarchical) :: chain
	integer :: i, j
	
! Set the observations
	chain%obs%K = 3
	chain%obs%r = 10
	chain%obs%V = 100.d0
	allocate(chain%obs%theta(chain%obs%K))
	allocate(chain%obs%y(chain%obs%K,chain%obs%r))
	
	do i = 1, chain%obs%K
		do j = 1, chain%obs%r
			chain%obs%y(i,j) = 5.d0*(i-1.d0) + 10.d0*randomn()
		enddo
	enddo

! Model
! mu ~ N(0,1)
! A ~ IG(1,1)
! V ~ IG(1,1)
! theta_i ~ Cauchy(mu,A)
! Y_ij ~ N(theta_i, V)

	call chain%initChain(chain%obs%K+3, 'METROPOLIS_GIBBS')

	do i = 1, 100000
		if (modulo(i, 1000) == 0) then
			print *, 'Iteration : ', i
		endif
		call chain%stepMGChain()
	enddo
	
! Do thinning and burn-in
	call chain%burninChain(10000)
	
	open(unit=12,file='posterior',action='write',status='replace',form='unformatted')
	write(12) chain%npar, chain%step
	write(12) chain%chain
	close(12)
	
end program mcmc_test