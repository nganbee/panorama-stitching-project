// File: v5_stages.h
// Authors: Team 8 (Thụ: V5 owner)
// Description: Device-resident stage interfaces for V5 (GPU data residency).
// External sources: none.
//
// V5 changes only the data path. Every function below launches the same
// kernels as the V1 stages (S1-S5) and the V3 texture S6, but takes and returns
// device pointers instead of host vectors. Intermediate images, keypoints,
// descriptors, matches and the homography never leave the GPU. Only the input
// RGB images go host->device, and only metadata needed for control (counts,
// status flags, canvas size, crop rectangle) plus the final panorama go
// device->host. All functions synchronize like V1 and throw std::runtime_error
// on CUDA failures; there is no CPU fallback.
#pragma once
#include "common.h"
#include <cstdint>

// Same layout as the private S5 result record (checked by static_assert in
// v1_stage5.cu). H maps B -> A.
struct V5HomographyResult {
    Homography H{};
    int inliers = 0;
    int ok = 0;
};

namespace v5 {
// S1: one thread per pixel, same formula/kernel as V1.
void rgb_to_gray(const uint8_t* rgb, int w, int h, float* gray);
// S1: Gaussian weights + separable blur. tmp and weights are caller scratch
// (w*h floats and 2*kGaussRadius+1 floats).
void gaussian_blur(const float* in, int w, int h, float* tmp, float* weights, float* out);

// S2: Sobel tensor + 5x5 Harris response. ixx/iyy/ixy are caller scratch.
void harris_response(const float* blurred, int w, int h,
                     float* ixx, float* iyy, float* ixy, float* response);
// S2: threshold + NMS + GPU merge sort. Writes at most kTopK sorted keypoints
// to `keypoints` (device, capacity kTopK) and returns their count. Only the
// candidate count is copied to the host.
int nms_topk(const float* response, int w, int h, Keypoint* keypoints);

// S3: BRIEF on device keypoints. Returns false when V1 would return an empty
// descriptor array (invalid keypoint); only the invalid flag is copied back.
bool describe(const float* blurred, int w, int h, const Keypoint* keypoints, int n,
              Descriptor* descriptors);

// S4: Hamming matching, ratio test, cross-check and ordered compaction.
// `matches` is device memory with capacity na. Returns the match count; only
// that count is copied back.
int match(const Descriptor* a, int na, const Descriptor* b, int nb, Match* matches);

// S5: RANSAC + refinement on device keypoints/matches. The result stays in
// `device_result`; a copy is returned in `host_result` for status/reporting.
bool ransac_homography(const Keypoint* kpA, int na, const Keypoint* kpB, int nb,
                       const Match* matches, int n,
                       V5HomographyResult* device_result, V5HomographyResult& host_result);

// S6 (V3 texture variant): reads A/B from the device buffers uploaded for S1
// and H from the device S5 result. Only canvas metadata, the crop rectangle and
// the cropped panorama are copied to the host.
void warp_blend(const uint8_t* A, int Aw, int Ah, const uint8_t* B, int Bw, int Bh,
                const V5HomographyResult* device_result, Image& pano);
} // namespace v5
