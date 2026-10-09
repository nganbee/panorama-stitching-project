// entry point of each pipeline version
#pragma once
#include "common.h"

// One full pipeline: left=A, right=B (B warped onto A's frame).
// Returns false on failure (e.g., too few matches).
using PipelineFn = bool (*)(const Image& A, const Image& B, PipelineResult& out);

bool run_v0(const Image& A, const Image& B, PipelineResult& out);  
// bool run_v1(...);  // basic CUDA
// bool run_v2(...);  // optimized 1
// bool run_v3(...);  // optimized 2

// Registry used by main.cu for "--version N"
inline PipelineFn get_version(int v) {
    switch (v) {
        case 0: return run_v0;
        // case 1: return run_v1;
        default: return nullptr;
    }
}