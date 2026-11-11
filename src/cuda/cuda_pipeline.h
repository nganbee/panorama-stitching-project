// File: cuda_pipeline.h
// Authors: Team 8
// Description: Version 1 CUDA pipeline entry points.
#pragma once

#include "../core_headers/common.h"

// V1 baseline: S1-S4 and S6 use the CUDA stage wrappers; S5 intentionally
// remains the deterministic V0 CPU RANSAC implementation for this milestone.
bool run_v1(const Image& A, const Image& B, PipelineResult& out);

// Smoke test used by the CLI. Detailed parity tests remain in tools/.
bool cuda_selftest(int version);
