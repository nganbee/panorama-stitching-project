// Independent V3 pipeline entry points. Enable with PANORAMA_WITH_V3.
#pragma once
#include "../core_headers/common.h"

// Reuse V1 S1-S5; execute V3 texture S6
bool run_v3(const Image& A, const Image& B, PipelineResult& out);
bool cuda_selftest_v3();
