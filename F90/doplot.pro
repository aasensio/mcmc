pro doplot
	openr,2,'posterior',/f77
	npar = 0L
	nstep = 0L
	readu,2,npar,nstep
	ch = dblarr(nstep,npar)
	readu,2,ch
	close,2
	
	!p.multi = [0,3,2]
	for i = 0, 5 do begin
		cgplot, ch[*,i], psym=3
	endfor
	stop
end