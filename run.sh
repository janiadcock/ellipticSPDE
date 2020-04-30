#!/bin/bash -eu

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

# We build the command line in a string before executing it, and it's very hard
# to get this to work if the executable name contains spaces, so we punt.
if [[ "$EXECUTABLE" != "${EXECUTABLE%[[:space:]]*}" ]]; then
    quit "Cannot handle spaces in executable name"
fi
# Command-line arguments are passed directly to the job script. We need to
# accept multiple arguments separated by whitespace, and pass them through the
# environment. It is very hard to properly handle spaces in arguments in this
# mode, so we punt.
for (( i = 1; i <= $#; i++ )); do
    if [[ "${!i}" != "${!i%[[:space:]]*}" ]]; then
        quit "Cannot handle spaces in command line arguments"
    fi
done
export ARGS=$@

export WALLTIME="$(printf "%02d:%02d:00" $((MINUTES/60)) $((MINUTES%60)))"

if [ "$(uname)" == "Darwin" ]; then
    export DYLD_LIBRARY_PATH="${DYLD_LIBRARY_PATH:-}:$LEGION_DIR/bindings/regent/"
    if [[ ! -z "${HDF_ROOT:-}" ]]; then
        export DYLD_LIBRARY_PATH="${DYLD_LIBRARY_PATH:-}:$HDF_ROOT/lib"
    fi
else
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$LEGION_DIR/bindings/regent/"
    if [[ ! -z "${HDF_ROOT:-}" ]]; then
        export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$HDF_ROOT/lib"
    fi
fi

# Make sure the number of requested ranks is divisible by the number of nodes.
export NUM_NODES=$(( NUM_RANKS / RANKS_PER_NODE ))
if (( NUM_RANKS % RANKS_PER_NODE > 0 )); then
    export NUM_NODES=$(( NUM_NODES + 1 ))
fi
export NUM_RANKS=$(( NUM_NODES * RANKS_PER_NODE ))

export LOCAL_RUN=0

export REALM_BACKTRACE=1

export LEGION_FREEZE_ON_ERROR="$DEBUG"

###############################################################################
# Machine-specific handling
###############################################################################

GROUP="${GROUP:-stanford}"
export QUEUE="${QUEUE:-pbatch}"
DEPS=
if [[ ! -z "$AFTER" ]]; then
    DEPS="-w done($AFTER)"
fi
bsub -G "$GROUP" \
    -nnodes "$NUM_NODES" -W "$MINUTES" -q "$QUEUE" $DEPS \
    /usr/workspace/adcock4/ellipticSPDE/lassen.lsf
