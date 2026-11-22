// V3 pipeline entry points. Enable with PANORAMA_WITH_V2 and PANORAMA_WITH_V3.
#pragma once
#include "../core_headers/common.h"

// Reuse V2 S1–S5 with V3 warping, blending and crop.
bool run_v3(const Image& A, const Image& B, PipelineResult& out);
bool cuda_selftest_v3();

// Internal shared orchestration: V4 substitutes only blur; S2-S6, profiling,
// transfer policy and error handling remain identical to this V3 baseline.
namespace panorama_detail {
using BlurFn = void (*)(const Gray&, Gray&);
bool run_v3_with_blur(const Image& A, const Image& B, PipelineResult& out,
                      BlurFn blur, int version);
}
