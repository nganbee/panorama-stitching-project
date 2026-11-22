#!/usr/bin/env bash
# Authors: Nguyen Duy Khanh; implementation assisted by OpenAI Codex.
# External sources: none. Run from the repository root on Colab/Linux.
set -euo pipefail
mkdir -p build out reports
flags=(-O2 -std=c++17 --fmad=false -DPANORAMA_WITH_CUDA -DPANORAMA_WITH_V2
       -Isrc/core_headers -Isrc/libraries)
pipeline=(src/integration/v0_pipeline.cpp
          src/stage_implementations/v0_stage12.cpp
          src/stage_implementations/v0_stage34.cpp
          src/stage_implementations/v0_stage5.cpp
          src/stage_implementations/v0_stage6.cpp
          src/stage_implementations/v1_stage12.cu
          src/stage_implementations/v1_stage34.cu
          src/stage_implementations/v1_stage5.cu
          src/stage_implementations/v1_stage6.cu
          src/stage_implementations/v2_stage4.cu
          src/cuda/cuda_pipeline.cu src/cuda/v2_pipeline.cu)
nvcc "${flags[@]}" src/integration/main.cpp "${pipeline[@]}" \
    src/libraries/stb_impl.cpp -o build/stitch_v2
nvcc "${flags[@]}" tools/run_stage34_v2_cuda_selftest.cpp \
    src/stage_implementations/v0_stage34.cpp \
    src/stage_implementations/v1_stage34.cu \
    src/stage_implementations/v2_stage4.cu -o build/stage34_v2_selftest
nvcc "${flags[@]}" tools/test_v2_pipeline.cpp "${pipeline[@]}" \
    -o build/test_v2_pipeline
