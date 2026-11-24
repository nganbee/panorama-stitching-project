// File: v5_pipeline.h
// Authors: Team 8 (Thụ: V5 owner)
// Description: V5 pipeline entry points. Enable with PANORAMA_WITH_V5.
// External sources: none.
#pragma once
#include "../core_headers/common.h"

// V5 = latest integrated kernels (V1 S1-S5 + V3 texture S6) with all
// intermediate buffers kept on the GPU. Only input/output and control
// metadata cross the PCIe bus.
bool run_v5(const Image& A, const Image& B, PipelineResult& out);

// CLI smoke test: V5 must reproduce V3 exactly on a synthetic pair while
// transferring fewer bytes.
bool cuda_selftest_v5();
