// V3 pipeline entry points. Enable with PANORAMA_WITH_V2 and PANORAMA_WITH_V3.
#pragma once
#include "../core_headers/common.h"

// Reuse V2 S1–S5 with V3 warping, blending and crop.
bool run_v3(const Image& A, const Image& B, PipelineResult& out);
bool cuda_selftest_v3();
