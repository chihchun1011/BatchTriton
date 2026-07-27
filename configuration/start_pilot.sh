#!/bin/bash
set -e

CORES=${TardisDroneCores:-1}
MEMORY=${TardisDroneMemory:-2048}
UUID=${TardisDroneUuid:-unknown}

exec apptainer exec --nv \
  --bind /cvmfs:/cvmfs \
  /cvmfs/unpacked.cern.ch/nvcr.io/nvidia/tritonserver:24.11-py3 \
  tritonserver \
    --model-repository=/cvmfs/cms.cern.ch/el9_amd64_gcc12/cms/cmssw/CMSSW_14_1_0_pre7/external/el9_amd64_gcc12/data/RecoBTag/Combined/data/models/ \ 
    --log-verbose=0 \
    --strict-model-config=false \
    --disable-auto-complete-config \
    --exit-timeout-secs=60