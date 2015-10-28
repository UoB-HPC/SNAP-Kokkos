#
# Build SNAP
#

default: build

# User defined parameters
DBG 	 	= no
PROFILER 	= yes
COPTION  	= 
FOPTION  	= 
OPENMP 	 	= -fopenmp
EXE 	 	= snap
TARGET	    = OMP
CXX	        = CC
FORTRAN  	= ftn
KOKKOS_PATH = ../kokkos_src

# Enables debugging mode
ifeq ($(DBG),yes)
  #EXE = snap_db
  FFLAGS = -g -O0 -check bounds -traceback  $(FOPTION)
  #FFLAGS = -O0 -g -C -Mchkstk -Ktrap=fp -Mchkfpstk -Mchkptr 
  CFLAGS = -g -O0 -debug all -traceback  $(COPTION)
  #CFLAGS = -O0 -g -C -Mchkstk -Ktrap=fp -Mchkfpstk 
else
  FFLAGS = -O3
  CFLAGS = -O3 $(COPTION) -std=c++11
endif

CFLAGS2 = -lm
FFLAG2 =

# Enables fine grained profiling of C implementation
ifeq ($(PROFILER), yes)
  CFLAGS += -DENABLE_PROFILING
endif

ifeq ($(TARGET), CUDA)
  CXX 		   = $(KOKKOS_PATH)/config/nvcc_wrapper
  KOKKOS_DEVICES = "Cuda"
  KOKKOS_ARCH	   = "Kepler35"
  CXXFLAGS 	  += -DCUDA
else
  KOKKOS_DEVICES = "OpenMP"
  KOKKOS_ARCH	   = "SNB"
endif

ifeq ($(TARGET), MIC)
  KOKKOS_ARCH = "KNC"
  FLAGS	  += -mmic
endif

ifdef TIMING
  CFLAGS += -DTIMING
endif

.SUFFIXES:	.f90 .o

EXT_OBJS = ext_sweep.o ext_core.o ext_kernels.o ext_profiler.o

OBJS = global.o snap_main.o utils.o version.o plib.o geom.o sn.o \
	   data.o control.o input.o setup.o dealloc.o translv.o solvar.o \
	   outer.o expxs.o inner.o sweep.o octsweep.o dim1_sweep.o dim3_sweep.o \
	   output.o time.o mms.o $(EXT_OBJS)

SRCS = global.f90 snap_main.f90 utils.f90 version.f90 plib.f90 geom.f90 \
	   sn.f90 data.f90 control.f90 input.f90 setup.f90 dealloc.f90 \
	   translv.f90 solvar.f90 outer.f90 expxs.f90 inner.f90 sweep.f90 \
	   octsweep.f90 dim1_sweep.f90 dim3_sweep.f90 output.f90 time.f90 mms.f90

ext_init.o:	ext_kernels.h
snap_main.o :	global.o utils.o version.o input.o setup.o output.o time.o \
	dealloc.o plib.o control.o
utils.o :	global.o dealloc.o plib.o
version.o :	global.o
plib.o :	global.o time.o
geom.o :	global.o
sn.o :	global.o
data.o :	global.o
control.o :	global.o
input.o :	global.o plib.o geom.o sn.o data.o control.o utils.o time.o
setup.o :	global.o geom.o sn.o data.o control.o mms.o utils.o time.o \
	plib.o
dealloc.o :	global.o sn.o data.o mms.o geom.o solvar.o control.o
translv.o :	global.o plib.o geom.o sn.o data.o control.o utils.o \
	solvar.o expxs.o outer.o time.o
solvar.o :	global.o plib.o geom.o sn.o data.o control.o
outer.o :	global.o geom.o sn.o data.o solvar.o control.o inner.o \
	time.o plib.o expxs.o
expxs.o :	global.o geom.o sn.o data.o
inner.o :	global.o geom.o sn.o data.o control.o solvar.o sweep.o \
	time.o plib.o
sweep.o :	global.o geom.o sn.o data.o control.o octsweep.o solvar.o
octsweep.o :	global.o geom.o sn.o data.o control.o solvar.o \
	dim1_sweep.o dim3_sweep.o
dim1_sweep.o :	global.o geom.o sn.o data.o control.o
dim3_sweep.o :	global.o plib.o geom.o sn.o data.o control.o
output.o :	global.o geom.o sn.o data.o control.o utils.o solvar.o \
	mms.o time.o plib.o
time.o :	global.o
mms.o:	global.o geom.o sn.o data.o control.o plib.o

include $(KOKKOS_PATH)/Makefile.kokkos

build: $(EXE)

$(EXE) : Makefile $(OBJS) $(KOKKOS_LINK_DEPENDS)
	$(FORTRAN) $(KOKKOS_LD_FLAGS) $(FFLAGS) $(FFLAG2) $(KOKKOS_LIBS) -o $@ $(OBJS)

#
# Fortran rule
#
%.o:	%.f90
	$(FORTRAN) $(FFLAGS) -c $<

#
# CPP rule
#
%.o:	%.cpp $(KOKKOS_CPP_DEPENDS) Makefile ext_sweep.hpp
	$(CXX) $(KOKKOS_CXXFLAGS) $(KOKKOS_CPPFLAGS) $(CXXFLAGS) -c $< $(CXXFLAGS2)

#
# Cleanup
#
clean: kokkos-clean
	rm -f *.o *.mod *.bc sweep_kernels.h

#
# Count lines of code
#
count:
	rm -f Lines
	for file in $(SRCS); do ./LineCount $$file Lines; done
	gawk -f ./LineReport < Lines >> Lines
	cat Lines

#
# Link compiled files only. No recompile.
#
link:
	$(FORTRAN) $(FFLAGS) $(FFLAG2) -o $(EXE) $(OBJS)

