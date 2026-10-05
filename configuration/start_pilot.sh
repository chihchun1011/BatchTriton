#!/bin/bash
set -e

CORES=${TardisDroneCores:-1}
MEMORY=${TardisDroneMemory:-2048}
GPUS=${TardisDroneGpus:-1}
UUID=${TardisDroneUuid:-unknown}

# Multiple pilots can land on the same GPU node, so each Triton instance
# needs its own ports instead of tritonserver's fixed defaults
# (8000/8001/8002), which would collide. 
# TritonAdapter._port_offset() in triton.py computes the same offset from
# the same UUID to know which port to query - keep the two in sync.
OFFSET=$(( 0x$(printf '%s' "$UUID" | md5sum | cut -c1-8) % 10000 ))
HTTP_PORT=$((8000 + OFFSET))
GRPC_PORT=$((8001 + OFFSET))
METRICS_PORT=$((8002 + OFFSET))

exec apptainer exec --nv \
--bind /cvmfs:/cvmfs \
--env LD_LIBRARY_PATH=/usr/local/cuda-12.9/targets/x86_64-linux/lib \
/cvmfs/unpacked.cern.ch/nvcr.io/nvidia/tritonserver:24.11-py3 \
tritonserver \
--model-repository=/cvmfs/cms.cern.ch/el9_amd64_gcc12/cms/cmssw/CMSSW_14_1_0_pre7/external/el9_amd64_gcc12/data/RecoBTag/Combined/data/models/ \
--http-port=${HTTP_PORT} \
--grpc-port=${GRPC_PORT} \
--metrics-port=${METRICS_PORT} \
--log-verbose=0 \
--strict-model-config=false \
--disable-auto-complete-config \
--exit-timeout-secs=60