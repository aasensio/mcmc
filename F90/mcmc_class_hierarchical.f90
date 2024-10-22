module mcmc_class_hierarchical
use mcmc_class
implicit none

	type observations
		integer :: K, r
		real(kind=8) :: V
		real(kind=8), allocatable :: theta(:), y(:,:)
	end type observations
	
	type, EXTENDS (mcmc) :: mcmc_hierarchical
	
		type(observations) :: obs
	
 		contains
 			procedure :: evalPosterior => logPosterior
 			procedure :: initialValues => initialValuesHierarchical
	end type mcmc_hierarchical

	contains

!------------------------------------------------------------------
! Set initial values for the parameters
!------------------------------------------------------------------
	subroutine initialValuesHierarchical(this)
	class(mcmc_hierarchical), intent(inout) :: this

		this%pars = 1.d0

	end subroutine initialValuesHierarchical

!------------------------------------------------------------------
! Evaluate the posterior. This function overrides the function in the parent
!------------------------------------------------------------------
	function logPosterior(this, outBounds) result (logP)
	class(mcmc_hierarchical), intent(in) :: this
	integer, intent(inout) :: outBounds
	real(kind=8) :: logP
	real(kind=8) :: mu, sigma, a, b
	integer :: i
	
! Parameters
! A  -> trial(1)
! V  -> trial(2)
! mu -> trial(3)
! theta -> trial(4:K+3)

		logP = 0.d0
		
!-----------------
! LOG-PRIORS
!-----------------

! A ~ IG(1,1)
		a = 1.d0
		b = 1.d0
		logP = logP - (a+1.d0)*log(this%trial(1)) - a / this%trial(1)
		
! V ~ IG(1,1)
		a = 1.d0
		b = 1.d0
		logP = logP - (a+1.d0)*log(this%trial(2)) - a / this%trial(2)
		
! mu ~ N(0,1)
		mu = 0.d0
		sigma = 1.d0
		logP = logP - (this%trial(3) - mu)**2 / (2.d0 * sigma**2) - log(sqrt(2.d0*PI)*sigma**2)
		
! theta ~ Cauchy(mu,A)
		do i = 1, this%obs%K
			logP = logP - log(this%trial(1)) - log(1.d0+((this%trial(3+i) - this%trial(3)) / this%trial(1))**2)
		enddo

!-----------------
! DATA LOG-LIKELIHOOD
!-----------------	
		do i = 1, this%obs%K
			logP = logP - 0.5d0 * this%obs%r * log(this%trial(2)) - sum( (this%obs%y(i,:) - this%trial(3+i))**2 ) / (2.d0*this%trial(2))
		enddo
		
		outBounds = 0
		
		return

	end function logPosterior

end module mcmc_class_hierarchical