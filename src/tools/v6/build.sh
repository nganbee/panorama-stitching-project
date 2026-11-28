#!/usr/bin/env bash
# Authors: Nguyen Duy Khanh; implementation assisted by OpenAI Codex.
# External sources: none. Run from the repository root on Colab/Linux.
set -euo pipefail
mkdir -p build out reports
flags=(-O2 -std=c++17 --fmad=false -DPANORAMA_WITH_CUDA -DPANORAMA_WITH_V2 -DPANORAMA_WITH_V3 -DPANORAMA_WITH_V4 -DPANORAMA_WITH_V5 -DPANORAMA_WITH_V6
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
          src/stage_implementations/v3_stage6.cu src/stage_implementations/v4_stage6.cu
          src/stage_implementations/v5_stage4.cu src/stage_implementations/v5_stage6.cu
          src/stage_implementations/v6_stage4.cu
          src/cuda/cuda_pipeline.cu src/cuda/v2_pipeline.cu src/cuda/v3_pipeline.cu
          src/cuda/v4_pipeline.cu src/cuda/v5_pipeline.cu src/cuda/v6_pipeline.cu)
nvcc "${flags[@]}" src/integration/main.cpp "${pipeline[@]}" \
    src/libraries/stb_impl.cpp -o build/stitch_v6
nvcc "${flags[@]}" src/tools/v6/test_matching.cu \
    src/stage_implementations/v0_stage34.cpp \
    src/stage_implementations/v5_stage4.cu -o build/stage4_v6_selftest
nvcc "${flags[@]}" src/tools/v6/test_pipeline.cpp "${pipeline[@]}" \
    src/libraries/stb_impl.cpp -o build/test_v6_pipeline
nvcc "${flags[@]}" src/tools/v6/benchmark_pair.cpp "${pipeline[@]}" \
    src/libraries/stb_impl.cpp -o build/benchmark_v6_pair
nvcc "${flags[@]}" src/tools/v6/verify_pair.cpp "${pipeline[@]}" \
    src/libraries/stb_impl.cpp -o build/verify_v6_pair
