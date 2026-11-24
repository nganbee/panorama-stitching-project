// File: cuda_pipeline.h
// Authors: Team 8
// Description: Version 1 CUDA pipeline entry points.
#pragma once

#include "../core_headers/common.h"

// V1 baseline: every S1-S6 computation executes in basic GPU kernels.
// CPU wrappers only validate buffer metadata, allocate, copy and launch.
bool run_v1(const Image& A, const Image& B, PipelineResult& out);

// Smoke test used by the CLI. Detailed parity tests remain in tools/.
bool cuda_selftest(int version);
