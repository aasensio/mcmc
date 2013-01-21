# Makefile for building the MCMC F90 code BayesME

# Select compiler
# Intel Fortran compiler
COMPILER = ifort
OPTIONS = -c -O3

LIBS=

# GFortran
#COMPILER = gfortran
#OPTIONS = -c -frecord-marker=4

OBJECTS = maths.o mcmc_class.o mcmc_class_hierarchical.o mcmc.o
	
mcmc: $(OBJECTS)
	$(COMPILER) $(OBJECTS) -o mcmc $(LIBS)	

clean:
	find . -maxdepth 3 -name "*.o" -delete ; find . -maxdepth 3 -name "*.mod" -delete
	find . -maxdepth 1 -name "*.f90~" -delete ; find . -maxdepth 1 -name "mcmc" -delete
	find ../ -maxdepth 1 -name "mcmc" -delete ; find . -maxdepth 2 -name "*.a" -delete
	
maths.o: maths.f90
	$(COMPILER) $(OPTIONS) maths.f90
	
mcmc_class.o: mcmc_class.f90 maths.o
	$(COMPILER) $(OPTIONS) mcmc_class.f90

mcmc_class_hierarchical.o: mcmc_class.o mcmc_class_hierarchical.f90
	$(COMPILER) $(OPTIONS) mcmc_class_hierarchical.f90

mcmc.o: mcmc.f90 mcmc_class.o mcmc_class_hierarchical.o
	$(COMPILER) $(OPTIONS) mcmc.f90
