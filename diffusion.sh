#!/bin/bash -eu

###############################################################################
# Inputs
###############################################################################

# Which group to submit jobs under (if a scheduler is available)
export GROUP="${GROUP:-}"

# Which queue/partition to use (if a scheduler is available)
export QUEUE="${QUEUE:-}"

# Which job to wait for before starting (if a scheduler is available)
export AFTER="${AFTER:-}"

# Whether to use GPUs (if available)
export USE_CUDA="${USE_CUDA:-1}"

# Whether to emit Legion profiler logs
export PROFILE="${PROFILE:-0}"

# Whether to freeze Legion execution on crash
export DEBUG="${DEBUG:-0}"

# How many ranks to instantiate per node
export RANKS_PER_NODE="${RANKS_PER_NODE:-1}"

# How many cores per rank to reserve for the runtime
export RESERVED_CORES="${RESERVED_CORES:-8}"

# Whether to dump additional HDF files, for debugging cross-section copying
export DEBUG_COPYING="${DEBUG_COPYING:-0}"

###############################################################################
# Helper functions
###############################################################################

function quit {
    echo "$1" >&2
    exit 1
}

###############################################################################
# Derived options
###############################################################################

export EXECUTABLE=/usr/workspace/adcock4/ellipticSPDE/diffusion.exec

# Total wall-clock time is the maximum across all samples.
# Total number of ranks is the sum of all sample rank requirements.

# see soleil.sh for how to define MINUTES, NUM_RANKS
# MINUTES=int(sample['Mapping']['wallTime'])
MINUTES=720

# tiles = sample['Mapping']['tiles']
# tilesPerRank = sample['Mapping']['tilesPerRank']
# xRanks = int(tiles[0]) / int(tilesPerRank[0])
# yRanks = int(tiles[1]) / int(tilesPerRank[1])
# zRanks = int(tiles[2]) / int(tilesPerRank[2])
# NUM_RANKS= xRanks * yRanks * zRanks
NUM_RANKS=1

export MINUTES
export NUM_RANKS

###############################################################################

source /usr/workspace/adcock4/ellipticSPDE/run.sh
