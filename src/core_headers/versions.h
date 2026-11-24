// File: versions.h
// Authors: Team 8; maintenance assisted by OpenAI Codex.
// Description: Pipeline registry; CUDA versions are compiled only when enabled.
// External sources: none.
#pragma once
#include "common.h"

#if defined(PANORAMA_WITH_V3) && (!defined(PANORAMA_WITH_CUDA) || !defined(PANORAMA_WITH_V2))
#error "PANORAMA_WITH_V3 requires PANORAMA_WITH_CUDA and PANORAMA_WITH_V2"
#endif

#if defined(PANORAMA_WITH_V4) && (!defined(PANORAMA_WITH_CUDA) || !defined(PANORAMA_WITH_V3))
#error "V4 requires PANORAMA_WITH_CUDA and PANORAMA_WITH_V3"
#endif

// One full pipeline: left=A, right=B (B warped onto A's frame).
// Returns false on failure (e.g., too few matches).
using PipelineFn = bool (*)(const Image& A, const Image& B, PipelineResult& out);

bool run_v0(const Image& A, const Image& B, PipelineResult& out);  
#ifdef PANORAMA_WITH_CUDA
#include "../cuda/cuda_pipeline.h"
#ifdef PANORAMA_WITH_V2
#include "../cuda/v2_pipeline.h"
#endif
#ifdef PANORAMA_WITH_V3
#include "../cuda/v3_pipeline.h"
#endif
#ifdef PANORAMA_WITH_V4
#include "../cuda/v4_pipeline.h"
#endif
#endif

// Registry used by main.cu for "--version N"
inline PipelineFn get_version(int v) {
    switch (v) {
        case 0: return run_v0;
#ifdef PANORAMA_WITH_CUDA
        case 1: return run_v1;
#ifdef PANORAMA_WITH_V2
        case 2: return run_v2;
#endif
#ifdef PANORAMA_WITH_V3
        case 3: return run_v3;
#endif
#ifdef PANORAMA_WITH_V4
        case 4: return run_v4; // Exact merged V3 baseline, optimizing S6 warp/blend
#endif
#endif
        default: return nullptr;
    }
}
