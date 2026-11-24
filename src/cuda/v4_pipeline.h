// V4 = merged V3 baseline with shared-memory Gaussian blur only.
#pragma once
#include "../core_headers/common.h"
bool run_v4(const Image& A, const Image& B, PipelineResult& out);
bool cuda_selftest_v4();
