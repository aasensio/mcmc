function mrandomn, seed, rmean, covar, nrand

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;+
;  NAME:
;     MRANDOMN
; PURPOSE:
; Function to draw NRAND random deviates from a multivariate normal
; distribution with zero mean and covariance matrix COVAR.
;
; AUTHOR : Brandon C. Kelly, Steward Obs., Sept. 2004
;
; INPUTS :
;
;    SEED - The random number generator seed, the default is IDL's
;           default in RANDOMN()
;    COVAR - The covariance matrix of the multivariate normal
;            distribution.
; OPTIONAL INPUTS :
;
;    NRAND - The number of randomn deviates to draw. The default is
;            one.
; OUTPUT :
;
;    The random deviates, an [NP,NRAND] array where NP is the
;    dimension of the covariance matrix, i.e., the number of
;    parameters.
;
; ROUTINES CALLED :
;
;    POSITIVE, DIAG
;-
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

	if n_params() lt 2 then begin
    	print, 'Syntax- Result = mrandomn( seed, covar, [nrand] )'
    	return, 0
	endif

;check inputs and set up defaults
	if n_elements(nrand) eq 0 then nrand = 1
	if size(covar, /n_dim) ne 2 then begin
    	return, covar*randomn(seed,nrand)+rmean
	endif

	np = (size(covar))[1]
	if (size(covar))[2] ne np then begin
    	print, 'COVAR must be a square matrix.'
    	return, 0
	endif

	diag = lindgen(np) * (np + 1L)
	epsilon = randomn(seed, nrand, np) ;standard normal random deviates (NP x NRAND matrix)

	A = covar + 1.d-8 * identity(np) ;store covariance into dummy variable
	                                  ;for input into TRIRED

	choldc, A, P, /double           ;do Cholesky decomposition

	for j = 0, np - 1 do for i = j, np - 1 do A[i,j] = 0d

	A[diag] = P

;transform standard normal deviates so they have covariance matrix COVAR

	epsilon = A ## epsilon + rmean##replicate(1.d0,nrand)

	return, transpose(epsilon)
end

function computeCholesky, covar
	np = n_elements(covar[*,0])
	diag = lindgen(np) * (np + 1L)

	A = covar + 1.d-8 * identity(np) ;store covariance into dummy variable
	                                  ;for input into TRIRED

	choldc, A, P, /double           ;do Cholesky decomposition

	for j = 0, np - 1 do for i = j, np - 1 do A[i,j] = 0d

	A[diag] = P

	return, A
end

function mrandomn_cholesky, seed, rmean, cholesky, nrand

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;+
;  NAME:
;     MRANDOMN
; PURPOSE:
; Function to draw NRAND random deviates from a multivariate normal
; distribution with zero mean and Cholesky decomposition of the covariance matrix cholesky.
;
; AUTHOR : Brandon C. Kelly, Steward Obs., Sept. 2004
;
; INPUTS :
;
;    SEED - The random number generator seed, the default is IDL's
;           default in RANDOMN()
;    COVAR - The covariance matrix of the multivariate normal
;            distribution.
; OPTIONAL INPUTS :
;
;    NRAND - The number of randomn deviates to draw. The default is
;            one.
; OUTPUT :
;
;    The random deviates, an [NP,NRAND] array where NP is the
;    dimension of the covariance matrix, i.e., the number of
;    parameters.
;
; ROUTINES CALLED :
;
;    POSITIVE, DIAG
;-
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


;check inputs and set up defaults
	if n_elements(nrand) eq 0 then nrand = 1

;transform standard normal deviates so they have covariance matrix COVAR
	np = n_elements(rmean)
	epsilon = randomn(seed, nrand, np) ;standard normal random deviates (NP x NRAND matrix)
	epsilon = cholesky ## epsilon + rmean##replicate(1.d0,nrand)

	return, transpose(epsilon)
end

; Sample from from the multivariate mixture of Gaussians (1-b)*N(mu=x,sigma_1) + b*N(mu=x,sigma_2)
function mixture_gaussians, seed, rmean1, covar1, rmean2, covar2, beta, nrand
	temp = randomu(seed,nrand)

	out = dblarr(n_elements(rmean1),nrand)
	
	ind = where(temp ge beta, count)
	if (count ne 0) then begin
		out[*,ind] = mrandomn(seed, rmean1, covar1, count)
	endif

	ind = where(temp lt beta, count)
	if (count ne 0) then begin
		out[*,ind] = mrandomn(seed, rmean2, covar2, count)
	endif

	return, out
end

pro test_random
	covar1 = [[1.d0,0.d0],[0.d0,1.d0]]
	covar2 = [[1.d0,0.d0],[0.d0,1.d0]]
	rmean1 = [0.d0,0.d0]
	rmean2 = [8.d0,0.d0]
	res=mixture_gaussians(seed, rmean1, covar1, rmean2, covar2, 0.2, 1000)
	stop
end