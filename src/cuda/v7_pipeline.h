// File: v7_pipeline.h
// Authors: Team 8 (Thụ: V7 owner)
// Description: V7 pipeline entry points. Enable with PANORAMA_WITH_V7.
// External sources: none.
#pragma once
#include "../core_headers/common.h"

// V7 = latest integrated pipeline (V5: V1 S1-S3, V2 S4 tiling, V3/V4 S6,
// GPU data residency) with S5 inlier scoring done by one warp per RANSAC
// hypothesis (v7_stage5.cu). Only S5 differs from V5.
bool run_v7(const Image& A, const Image& B, PipelineResult& out);

// CLI smoke test: V7 must reproduce V5 bit for bit on a synthetic pair
// (same keypoints, matches, inliers, H and panorama bytes).
bool cuda_selftest_v7();
