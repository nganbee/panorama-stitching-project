// File: versions.h
// Authors: Team 8; maintenance assisted by OpenAI Codex.
// Description: Pipeline registry; CUDA versions are compiled only when enabled.
// External sources: none.
#pragma once
#include "common.h"

// One full pipeline: left=A, right=B (B warped onto A's frame).
// Returns false on failure (e.g., too few matches).
using PipelineFn = bool (*)(const Image& A, const Image& B, PipelineResult& out);

bool run_v0(const Image& A, const Image& B, PipelineResult& out);  
#ifdef PANORAMA_WITH_CUDA
#include "../cuda/cuda_pipeline.h"
#ifdef PANORAMA_WITH_V3
#include "../cuda/v3_pipeline.h"
#ifdef PANORAMA_WITH_V5
#include "../cuda/v5_pipeline.h"
#endif
#endif
#endif

// Registry used by main.cu for "--version N"
inline PipelineFn get_version(int v) {
    switch (v) {
        case 0: return run_v0;
#ifdef PANORAMA_WITH_CUDA
        case 1: return run_v1;
#ifdef PANORAMA_WITH_V3
        case 3: return run_v3; // V1 S1-S5 + V3 S6; V2 integration is pending
#ifdef PANORAMA_WITH_V5
        case 5: return run_v5; // V3 kernels + GPU data residency; V2/V4 pending
#endif
#endif
#endif
        default: return nullptr;
    }
}
