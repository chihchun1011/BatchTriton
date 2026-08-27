#!/bin/bash
set -e

CORES=${TardisDroneCores:-1}
MEMORY=${TardisDroneMemory:-2048}
GPUS=${TardisDroneGpus:-1} 
UUID=${TardisDroneUuid:-unknown}

exec apptainer exec --nv \
--bind /cvmfs:/cvmfs \
--env LD_LIBRARY_PATH=/usr/local/cuda-12.9/targets/x86_64-linux/lib \
/cvmfs/unpacked.cern.ch/nvcr.io/nvidia/tritonserver:24.11-py3 \
tritonserver \
--model-repository=/cvmfs/cms.cern.ch/el9_amd64_gcc12/cms/cmssw/CMSSW_14_1_0_pre7/external/el9_amd64_gcc12/data/RecoBTag/Combined/data/models/ \
--log-verbose=0 \
--strict-model-config=false \
--disable-auto-complete-config \
--exit-timeout-secs=60