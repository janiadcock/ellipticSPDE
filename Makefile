# Required paths
ifndef LEGION_DIR
  $(error LEGION_DIR is not set)
endif

# OS-specific options
ifeq ($(shell uname),Darwin)
  DYNLINK_PATH := DYLD_LIBRARY_PATH
else
  DYNLINK_PATH := LD_LIBRARY_PATH
endif

# CUDA options
USE_CUDA ?= 1

# HDF options
export USE_HDF ?= 1
export HDF_HEADER ?= hdf5.h
HDF_LIBNAME ?= hdf5

# C compiler options
CFLAGS += -g -O2 -Wall -Werror -fno-strict-aliasing -I$(LEGION_DIR)/runtime -I$(LEGION_DIR)/bindings/regent
CXXFLAGS += -std=c++11 -g -O2 -Wall -Werror -fno-strict-aliasing -I$(LEGION_DIR)/runtime -I$(LEGION_DIR)/bindings/regent

# Regent options
export INCLUDE_PATH := .
ifdef HDF_ROOT
  export INCLUDE_PATH := $(INCLUDE_PATH);$(HDF_ROOT)/include
  export $(DYNLINK_PATH) := $($(DYNLINK_PATH)):$(HDF_ROOT)/lib
endif
REGENT := $(LEGION_DIR)/language/regent.py -g
ifeq ($(USE_CUDA), 1)
  REGENT_FLAGS := -fflow 0 -fopenmp 1 -foverride-demand-openmp 1 -finner 1 -fcuda 1 -fcuda-offline 1 -foverride-demand-cuda 1
else
  REGENT_FLAGS := -fflow 0 -fopenmp 1 -foverride-demand-openmp 1 -finner 1 -fcuda 0 -foverride-demand-cuda 1
endif

# Link flags
ifdef CRAYPE_VERSION
  LINK_FLAGS += -Bdynamic
  LINK_FLAGS += $(CRAY_UGNI_POST_LINK_OPTS) -lugni
  LINK_FLAGS += $(CRAY_UDREG_POST_LINK_OPTS) -ludreg
endif
LINK_FLAGS += -L$(LEGION_DIR)/bindings/regent -lregent
ifdef HDF_ROOT
  LINK_FLAGS += -L$(HDF_ROOT)/lib
endif
ifeq ($(USE_HDF), 1)
  LINK_FLAGS += -l$(HDF_LIBNAME)
endif
LINK_FLAGS += -lm

.PHONY: default all clean

default: diffusion.exec

all: diffusion.exec

clean:
	$(RM) *.exec *.o 

%-desugared.rg: %.rg
	./desugar.py $< > $@

diffusion.exec: diffusion.o 
	$(CXX) -o $@ $^ $(LINK_FLAGS)

diffusion.o: diffusion-desugared.rg
	$(REGENT) diffusion-desugared.rg $(REGENT_FLAGS)
