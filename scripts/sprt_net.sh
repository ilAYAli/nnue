#! /bin/bash

if (( $# < 2 || $# > 3 )); then
    echo "usage: $0 REFERENCE_NET CANDIDATE_NET [ENGINE]" >&2
    exit 2
fi

#CANDIDATE_NET="~/assets/nets/enyo-scratch-long-1.5.0-rc1.nn"
#REFERENCE_NET="~/assets/nets/default.net"
REFERENCE_NET=$1
CANDIDATE_NET=$2
#ENGINE=${3:-"~/assets/engines/enyo_7483c1c"}
ENGINE=${3:-"~/assets/engines/candidate"}

set +x
forge sprt \
    --candidate "$ENGINE" \
    --candidate-uci nnue_file="$CANDIDATE_NET" \
    --reference "$ENGINE" \
    --reference-uci nnue_file="$REFERENCE_NET" \
    --restart on \
    --hash 128 \
    --shards 60 \
    --elo1 10 \
    --alpha 1e-300 \
    --beta 1e-300 \
    --games 1000
