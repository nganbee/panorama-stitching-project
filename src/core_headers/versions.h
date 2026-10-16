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
#endif

// Registry used by main.cu for "--version N"
inline PipelineFn get_version(int v) {
    switch (v) {
        case 0: return run_v0;
#ifdef PANORAMA_WITH_CUDA
        case 1: return run_v1;
        case 2: return run_v2;
        case 3: return run_v3;
#endif
        default: return nullptr;
    }
}
