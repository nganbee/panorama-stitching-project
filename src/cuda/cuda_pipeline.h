// Authors: Team 8, with Codex implementation assistance. External sources: none.
#pragma once
#include "../core_headers/common.h"

// Real CUDA pipelines: the host retains deterministic top-K selection,
// hypothesis sampling, final least-squares refinement and valid-mask cropping.
// All failures reset the panorama and return false; diagnostics go to stderr.
bool run_v1(const Image& A, const Image& B, PipelineResult& out);
bool run_v2(const Image& A, const Image& B, PipelineResult& out);
bool run_v3(const Image& A, const Image& B, PipelineResult& out);

// Executed only on an actual CUDA device. Compares device stages to the V0
// reference, with a known homography, and returns false for any discrepancy.
bool cuda_selftest(int version);
