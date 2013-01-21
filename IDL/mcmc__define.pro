@random_numbers
@vector

;***************************************
; Initialize the MCMC object
; npar: number of free parameters
; priors: a structure that gives information about the priors containing the following keys
;        left: dblarr(npar)  -> left border of the prior
;        right: dblarr(npar) -> right border of the prior
;        type: strarr(npar)  -> prior type ('U', 'G', 'D', 'J')
;        mu: dblarr(npar)    -> mean of the prior distribution if Gaussian
;        sigma: dblarr(npar)    -> standard deviation of the prior distribution if Gaussian
;        
; data: any structure that will be passed to the evaluation of the posterior distribution function
; adaptive_metropolis: initialize an adaptive Metropolis chain
; metropolis_within_gibbs: initialize an adaptive Metropolis-within-Gibbs chain
;***************************************
function mcmc::init, npar, priors, data=data, adaptive_metropolis=adaptive_metropolis, metropolis_within_gibbs=metropolis_within_gibbs
compile_opt idl2
on_error, 2

	self.data = ptr_new(data)
	self.npar = ptr_new(npar)
	self.pars = ptr_new(dblarr(npar))
	self.chain = ptr_new(dblarr(1,npar))
	self.mean = ptr_new(dblarr(npar))
	self.covariance = ptr_new(dblarr(npar,npar))
	self.identity = ptr_new(identity(npar,/double))
	self.trial = ptr_new(dblarr(npar))
	self.step = ptr_new(0)
	self.stepaccepted = ptr_new(0)
	self.prior = ptr_new(priors)
	self.logp = ptr_new(0.d0)
	self.logptrial = ptr_new(0.d0)
	self.alpha = ptr_new(0.d0)

	self.algorithm = ptr_new('MCMC')
	
	if (keyword_set(adaptive_metropolis)) then begin
		self.algorithm = ptr_new('ADAPTIVE')
	endif
	if (keyword_set(metropolis_within_gibbs)) then begin
		self.algorithm = ptr_new('METROPOLIS_GIBBS')
		self.ls = ptr_new(dblarr(npar))
		self.lsaccepted = ptr_new(dblarr(npar))
		self.steps_in_batch = ptr_new(0)
	endif
	
	return, 1
end

;***************************************
; Free memory after destroying the object
;***************************************
function mcmc::cleanup
	ptr_free, self.data
	ptr_free, self.npar
	ptr_free, self.pars
	ptr_free, self.mean
	ptr_free, self.covariance
	ptr_free, self.trial
	ptr_free, self.step
	ptr_free, self.stepaccepted
	ptr_free, self.logp
	ptr_free, self.logptrial
	ptr_free, self.alpha
	ptr_free, self.identity
	ptr_free, self.algorithm
	if (ptr_valid(self.ls)) then ptr_free, self.ls
	if (ptr_valid(self.lsaccepted)) then ptr_free, self.lsaccepted
	if (ptr_valid(self.steps_in_batch)) then ptr_free, self.steps_in_batch
	return, 1
end 

;***************************************
; Functions to get the parameters from the Markov chain
;***************************************
function mcmc::getPars
	return, (*self.pars)
end

;***************************************
; Functions to get the log of the standard deviation for the Metropolis-within-Gibbs method
;***************************************
function mcmc::getls
	return, (*self.ls)
end

;***************************************
; Functions to get the mean of the parameters from the Markov chain
;***************************************
function mcmc::getMean	
	return, average((*self.chain),1)
end

;***************************************
; Functions to get the mean of the parameters from the Markov chain
;***************************************
function mcmc::getStddev
	return, sig_array((*self.chain),1)
end

;***************************************
; Functions to get the acceptance rate from the Markov chain
;***************************************
function mcmc::getAcceptanceRate
	return, 1.d0*(*self.stepaccepted) / (*self.step)
end

;***************************************
; Functions to get the Markov chain
;***************************************
function mcmc::getChain
	return, (*self.chain)
end

;***************************************
; Functions to get the log-standard deviation of the Metropolis-within-Gibbs acceptance rate from the Markov chain
;***************************************
function mcmc::getlsAcceptanceRate, i
	return, 1.d0*(*self.lsaccepted)[i] / (*self.step)
end

;***************************************
; Functions to thin the chain
;***************************************
pro mcmc::thinChain, factor	
   new_length = (*self.step) / factor
   b = congrid( (*self.chain), new_length, (*self.npar) )
   ptr_free, self.chain
   self.chain = ptr_new(b)
   (*self.step) = new_length
end

;***************************************
; Functions to remove burn-in from the chain
;***************************************
pro mcmc::burnin, percent
	ind = fix((*self.step) * percent/100.d0)
	b = (*self.chain)[ind:*,*]
	ptr_free, self.chain
   self.chain = ptr_new(b)
   (*self.step) = n_elements(b[*,0])
end

;***************************************
; Functions to set the data structure in the Markov chain
;***************************************
pro mcmc::setData, data
	if (ptr_valid((*self.data))) then begin
		ptr_free, (*self.data)
	endif
	self.data = ptr_new(data)
end

;***************************************
; Functions to change the prior structure in the Markov chain
;***************************************
pro mcmc::setPrior, prior
	if (ptr_valid((*self.prior))) then begin
		ptr_free, (*self.prior)
	endif
	self.prior = ptr_new(prior)
end

;***************************************
; Update the statistics of the chain after a step
; reset: reset the statistics before beginning
;***************************************
pro mcmc::update_stat, reset=reset
		
	n = (*self.step)
	
; Update the mean
	mold = (*self.mean)
	(*self.mean) = mold + ( (*self.pars) - mold) / (n+1.d0)
		
; Update the covariance
	for i = 0, (*self.npar)-1 do begin
		for j = 0, (*self.npar)-1 do begin
			(*self.covariance)[i,j] = (n-1.d0)/n * (*self.covariance)[i,j] + $
				( (*self.pars)[i]-mold[i] ) * ( (*self.pars)[j] - mold[j] ) / $
				(n+1.d0)^2 + ( (*self.pars)[i] - (*self.mean)[i] )*$
				( (*self.pars)[j] - (*self.mean)[j] ) / (n*1.d0)
		endfor		
	endfor
end

;***************************************
; Select a new trial point
; In this case, use a multidimensional Gaussian distribution with mean
; equal to the current point and covariance equal to the estimated covariance
;***************************************
function mcmc::selectTrial

; Standard MCMC with Normal proposal that uses an adaptive covariance matrix
	if ((*self.algorithm) eq 'MCMC') then begin
		out = mrandomn(seed, (*self.pars), (*self.alpha) * (*self.covariance), 1)
	endif

; Adaptive Metropolis with mixture of Gaussians
	if ((*self.algorithm) eq 'ADAPTIVE') then begin

; If iteration is smaller than 2*d
		if ((*self.step) le 2*(*self.npar)) then begin
		
; Generate a random number from the Gaussians N(mu=x,sigma^2=0.1^2*I/d)
			out = mrandomn(seed, (*self.pars), 0.1^2 / (*self.npar) * (*self.identity), 1)
		endif else begin
		
; Generate a random number from the mixture of Gaussians (1-b)*N(mu=x,sigma^2=2.38^2*Sigma/d) + b*N(mu=x,sigma^2=0.1^2*I/d)
			covar1 = 2.38^2 * (*self.covariance) / (*self.npar)
			covar2 = 0.1^2 * (*self.identity) / (*self.npar)
			out = mixture_gaussians(seed, (*self.pars), covar1, (*self.pars), covar2, 0.05, 1)
		endelse

	endif

; Adaptive Metropolis-within-Gibbs
; Draw npar random numbers from N(0,sigma^2), with sigma appropriate for each parameter
	if ((*self.algorithm) eq 'METROPOLIS_GIBBS') then begin
		sigma = exp(*self.ls)
		out = (*self.pars) + sigma*randomn(seed,(*self.npar))
	endif
	
	return, out
end

;***************************************
; Evaluate the priors at the present point (log-priors)
; trial: value of the trial parameters
; outBounds: set this to 1 if the proposed model is outside the boundaries
;***************************************
function mcmc::evalPrior, trial, outBounds=outBounds
	
	logprior = 0.d0
	outBounds = 0
	
	for i = 0, (*self.npar)-1 do begin

; Dirac delta prior
		if ((*self.prior).type[i] eq 'DIRAC') then begin
			trial[i] = (*self.prior).mu[i]
		endif
		
; Verify that everything is inside bounds
		if (trial[i] gt (*self.prior).right[i] or trial[i] lt (*self.prior).left[i]) then begin
			outBounds = 1
			return, 0
		endif

; Gaussian prior p=1/sqrt(2*pi*sigma) * exp(-(x-mu)^2 / (2*sigma^2))
		if ((*self.prior).type[i] eq 'NORMAL') then begin
			mu = (*self.prior).mu[i]
			sigma = (*self.prior).sigma[i]
			logprior = logprior - (trial[i] - mu)^2 / (2.d0 * sigma^2) - alog(sqrt(2.d0*!DPI)*sigma^2)
		endif

; Maxwell prior p=2/sqrt(pi) x^2/sigma^3 * exp(-x^2 / (2*sigma^2))
		if ((*self.prior).type[i] eq 'MAXWELL') then begin
			sigma = (*self.prior).sigma[i]
			logprior = logprior + 2.d0 * alog(trial[i]) - 3.d0 * alog(sigma) - trial[i]^2 / (2.d0*sigma^2)
		endif

; logNormal prior
		if ((*self.prior).type[i] eq 'LOGNORMAL') then begin
			logprior = logprior - (alog(trial[i]) - (*self.prior).mu[i])^2 / (2.d0 * (*self.prior).sigma[i]^2) - $
				alog(trial[i]) - alog(sqrt(2.d0*!DPI)*(*self.prior).sigma[i]^2)
		endif

; Beta prior
		if ((*self.prior).type[i] eq 'BETA') then begin
			logprior = logprior + ( (*self.prior).alpha[i]-1.d0 ) * alog(trial[i]-(*self.prior).left[i]) + $
				( (*self.prior).beta[i]-1.d0 ) * alog((*self.prior).right[i] - trial[i])
		endif

; Jeffrey's prior
		if ((*self.prior).type[i] eq 'JEFFREYS') then begin
			logprior = logprior - alog(trial[i]) - alog( alog((*self.prior).right[i]) - alog((*self.prior).left[i]) )
		endif

; Gamma prior p=b^a * x^(a-1) * exp(-b*x)
		if ((*self.prior).type[i] eq 'GAMMA') then begin
			a = (*self.prior).mu[i]
			b = (*self.prior).sigma[i]
			logprior = logprior + (a-1.d0)*alog(trial[i]) - b*trial[i]
		endif

; Inverse Gamma prior p=b^a * x^-(a+1) * exp(-b/x)
		if ((*self.prior).type[i] eq 'IG') then begin
			a = (*self.prior).mu[i]
			b = (*self.prior).sigma[i]
 			logprior = logprior - (a+1.d0)*alog(trial[i]) - b/trial[i]
		endif

; Translated Cauchy prior p=1/(1+((x-m)/s)^2)
		if ((*self.prior).type[i] eq 'CAUCHY') then begin
			mu = (*self.prior).mu[i]
			sigma = (*self.prior).sigma[i]
			logprior = logprior - alog(1.d0+((trial[i]-mu)/sigma)^2)
		endif
		
	endfor
		
	return, logprior
end

;***************************************
; Evaluate the log-likelihood at the present point (a simple Gaussian likelihood in this case)
; trial: value of the trial parameters
; outBounds: set this to 1 if the proposed model is outside the boundaries
;***************************************
function mcmc::evalTarget, trial, outBounds=outBounds

; Evaluate the prior
	logprior = self->evalPrior(trial, outBounds=outBounds)
	
; If the point is outside the bounds, return 0
	if (logprior eq 0 and outBounds eq 1) then begin
		logposterior = 0.d0
		return, logposterior
	endif

;-------------------------------------------------------------------
; Evaluate likelihood. We choose a normal likelihood for a linear fit
;-------------------------------------------------------------------
; 	a = trial[0]
; 	b = trial[1]
; 	model = a * (*self.data).x + b
; 	chi2 = total( ((*self.data).y - model)^2 / ( (*self.data).sigma^2 ))


;-------------------------------------------------------------------
; Evaluate likelihood. We choose a normal likelihood for Metropolis-within-Gibbs problem
;-------------------------------------------------------------------
	K = (*self.data).K

	A = trial[0]
	V = trial[1]
	mu = trial[2]
	theta = trial[3:*]

	loghyper = 0.d0
	for i = 0, K-1 do begin		
		loghyper = loghyper - alog(A) - alog(1.d0+((theta[i]-mu)/A)^2)
	endfor

	logL = 0.d0
	for i = 0, K-1 do begin
		n = (*self.data).r
		Y = (*self.data).y[i,*]
		logL = logL - 0.5d0 * n * alog(V) - total( (Y - theta[i])^2 ) / (2.d0*V)
	endfor
			
	logposterior = logL + logprior + loghyper
	
	return, logposterior
end

;***************************************
; Do some initial initialization of the Markov Chain
;***************************************
function mcmc::initChain

; Initial point is chosen randomly from the available space
	(*self.pars) = randomu(seed,(*self.npar)) * ( (*self.prior).right - (*self.prior).left) + (*self.prior).left
	(*self.trial) = (*self.pars)
	(*self.mean) = (*self.pars)

	(*self.chain)[0,*] = (*self.pars)
	
; Initial covariance matrix is chosen to be 10% of the available width of the hyperspace
	for i = 0, (*self.npar)-1 do begin
		(*self.covariance)[i,i] = 0.1d0 * ((*self.prior).right[i] - (*self.prior).left[i])
	endfor
	
	(*self.logp) = self->evalTarget((*self.pars))
	
	(*self.alpha) = 1.d0

end


;***************************************
; Do a step of the Markov Chain
;***************************************
function mcmc::stepChain

; Add one step to the chain counter
	(*self.step) = (*self.step) + 1
	logp = (*self.logp)
	
; Propose a new set of parameters
	trial = self->selectTrial()

; Evaluate the posterior
	logptrial = self->evalTarget(trial, outBounds=outBounds)
	
; If it is inside the boundaries
	if (outBounds eq 0) then begin
	
; Metropolis-Hastings step verification
		r = exp(logptrial - logp)
		alpha = min([1.d0,r])
		ran = randomu(seed,1)
		
		
		if (ran lt alpha) then begin
			(*self.pars) = trial		
			(*self.logp) = logptrial
			
			(*self.stepaccepted) = (*self.stepaccepted) + 1
		endif
	endif
	
	acceptance = self->getAcceptanceRate()
	
; Modify the scaling factor of the proposal distribution to adapt the acceptance rate
	if ((*self.step) / 100 eq (*self.step) / 100.d0 and (*self.alpha) gt 0.05) then begin
		if (acceptance gt 0.4d0 + 0.05d0) then begin
			(*self.alpha) = (*self.alpha) * 1.04d0
		endif
		if (acceptance lt 0.4d0 - 0.05d0) then begin
			(*self.alpha) = (*self.alpha) * 0.96d0
		endif		
	endif
	
	if ((*self.step) gt 100) then begin
  		self->update_stat
	endif

; Update the chain
	chain = (*self.chain)
	chain = [chain, transpose((*self.pars))]

	ptr_free, self.chain
	self.chain = ptr_new(chain)

end


;***************************************
; Do a step of the Metropolis-within-Gibbs chain
;***************************************
function mcmc::stepChain_metropolis_gibbs

; Add one step to the chain counters
	(*self.step) = (*self.step) + 1
	(*self.steps_in_batch) = (*self.steps_in_batch) + 1
	reset_batch = 0
	
	logp = (*self.logp)

; Save the original set of parameters	
	updated_pars = (*self.pars)
	old_pars = (*self.pars)

; Propose a new set of parameters
	trial = self->selectTrial()

; Do the Gibbs updating
	for i = 0, (*self.npar)-1 do begin

; Set the i-th parameter to its new value and do a Metropolis step
		updated_pars[i] = trial[i]

; Evaluate the posterior
		logptrial = self->evalTarget(updated_pars, outBounds=outBounds)

; If it is inside the boundaries
		if (outBounds eq 0) then begin

; Metropolis-Hastings step verification
			r = exp(logptrial - logp)
			alpha = min([1.d0,r])
			ran = randomu(seed,1)

; If the MH step is successful, we keep the modification of the i-th parameter
			if (ran lt alpha) then begin				
				(*self.pars)[i] = updated_pars[i]
 				(*self.lsaccepted)[i] = (*self.lsaccepted)[i] + 1
 				logp = logptrial
			endif else begin
; If not, return the original value of the parameter
				updated_pars[i] = old_pars[i]
			endelse
		endif else begin
; If not, return the original value of the parameter
			updated_pars[i] = old_pars[i]
		endelse

; Modify the logarithm of the variances to adapt the acceptance rate
		if ( ((*self.step)+1) mod 50 eq 0) then begin
			acceptance = 1.d0*(*self.lsaccepted)[i] / (*self.steps_in_batch)
			n_batch = (*self.step) / 50.d0

			delta_n = min([0.01d0, 1.d0 / sqrt(n_batch)])

;  			if (i eq 3) then print, acceptance, delta_n, (*self.ls)[i]
			
			if (acceptance gt 0.44d0) then begin
				(*self.ls)[i] = (*self.ls)[i] + delta_n
			endif
			if (acceptance lt 0.44d0) then begin
				(*self.ls)[i] = (*self.ls)[i] - delta_n
			endif
			
			reset_batch = 1
		endif

	endfor

	(*self.logp) = logp

; Update the chain	
	chain = (*self.chain)
	chain = [chain, transpose((*self.pars))]
	
	ptr_free, self.chain
	self.chain = ptr_new(chain)

; The next iteration starts a new batch of 50 steps
	if (reset_batch eq 1) then begin
		(*self.lsaccepted) = (*self.lsaccepted) * 0
		(*self.steps_in_batch) = 0		
	endif

end

pro mcmc__define

	data = {mcmc, $
		data : ptr_new(), $
		npar : ptr_new(), $
		pars : ptr_new(), $
		mean : ptr_new(), $
		covariance : ptr_new(), $
		trial : ptr_new(), $
		step : ptr_new(), $
		prior : ptr_new(),$
		logp: ptr_new(), $
		logptrial: ptr_new(), $
		stepaccepted: ptr_new(), $
		alpha: ptr_new(),$
		identity: ptr_new(),$
		algorithm: ptr_new(),$
		ls: ptr_new(),$
		lsaccepted: ptr_new(),$
		chain: ptr_new(),$
		steps_in_batch: ptr_new()}
	
	return
	
end


pro test_mcmc

; Generate a linear function and add some noise
	x = findgen(20) / 19.d0 * 2.d0
	y = 2.34*x - 1.05
	noise = 0.05d0
	ynoise = y + noise*randomn(seed,20)
	
; Generate the structure with the data we need to evaluate the posterior
	data = {x: x, y: ynoise, sigma: replicate(noise,20)}
	
; Define the priors
	prior = {left: [1.d0,-2.d0], right: [4.d0,0.d0], type: ['UNIFORM','UNIFORM']}
	
; Create the MCMC object
	a = obj_new('mcmc', 2, prior, data=data, /adaptive_metropolis)
	
; Initialize the Markov chain
	res = a->initChain()
	
; Do 20000 steps of the chain
	nchain = 20000
	pars = dblarr(2,nchain)
	
	for i = 0, nchain-1 do begin
		res = a->stepChain()
		pars[*,i] = a->getPars()
		if (i mod 1000 eq 0) then begin
 			print, i, a->getAcceptanceRate()
		endif
	endfor
	
	!p.multi = [0,2,2]
	cgplot, pars[0,*], psym=3, tit='Mean='+strtrim(string(mean(pars[0,*])),2)+' - sigma='+strtrim(string(stddev(pars[0,*])),2)
	cghistoplot, pars[0,*]
	cgplot, pars[1,*], psym=3, tit='Mean='+strtrim(string(mean(pars[1,*])),2)+' - sigma='+strtrim(string(stddev(pars[1,*])),2)
	cghistoplot, pars[1,*]
	!p.multi = 0
	
	stop
	
end


pro test_metropolis_gibbs

	K = 3
	r = replicate(10, K)
	V = 10.d0^2
	theta = dindgen(K)
	y = dblarr(K,10)

; Generate Y_ij = N(i-1,10^2)
	for i = 0, K-1 do begin
		y[i,*] = 5.d0*i + 10.d0*randomn(seed,r[i])
	endfor

; Model
; mu ~ N(0,1)
; A ~ IG(1,1)
; V ~ IG(1,1)
; theta_i ~ Cauchy(mu,A)
; Y_ij ~ N(theta_i, V)

; Generate the structure with the data we need to evaluate the posterior
	data = {y: y, K: K, r: r}

; Define the priors
	left = dblarr(K+3)
	right = dblarr(K+3)
	mu = dblarr(K+3)
	sigma = dblarr(K+3)
	type = strarr(K+3)
	params = ['A','V','mu','th1','th2','th3']

; A
	left[0] = 0.d0
	right[0] = 200.d0
	mu[0] = 1.d0
	sigma[0] = 1.d0
	type[0] = 'IG'

; V
	left[1] = 0.d0
	right[1] = 200.d0
	mu[1] = 1.d0
	sigma[1] = 1.d0
	type[1] = 'IG'

; mu
	left[2] = -10.d0
	right[2] = 10.d0
	mu[2] = 0.d0
	sigma[2] = 1.d0
	type[2] = 'NORMAL'

; thetas
	left[3:*] = replicate(-200.d0,K)
	right[3:*] = replicate(200.d0,K)
	type[3:*] = replicate('UNIFORM',K)
	
	prior = {left: left, right: right, type: type, mu: mu, sigma: sigma}

; Create the MCMC object
	a = obj_new('mcmc', 3+K, prior, data=data, /metropolis_within_gibbs)

; Initialize the Markov chain
	res = a->initChain()

; Do 20000 steps of the chain
	nchain = 50000L
	pars = dblarr(K+3,nchain)
	ls = dblarr(K+3,nchain)

	for i = 0, nchain-1 do begin
		res = a->stepChain_metropolis_gibbs()
		pars[*,i] = a->getPars()
		ls[*,i] = a->getls()
		if (i mod 1000 eq 0) then begin
			print, 'Iteration : ', i
		endif
	endfor

; Burn-in and thinning
 	a->burnin, 20
 	a->thinChain, 2
	pars = a->getChain()

	mean_pars = a->getMean()
	sigma_pars = a->getStddev()

	cgdisplay,xsize=1000,ysize=800
	
	!p.multi = [0,4,3]
	for i = 0, 5 do cgplot, ls[i,*], psym=3, tit='Mean='+strtrim(string(mean(ls[i,*])),2)+' - sigma='+strtrim(string(stddev(ls[i,*])),2),$
		ytit=params[i]
	for i = 0, 5 do cgplot, pars[*,i], psym=3, tit='Mean='+strtrim(string(mean(pars[*,i])),2)+' - sigma='+strtrim(string(stddev(pars[*,i])),2),$
		ytit=params[i]
	!p.multi = 0

	stop

end