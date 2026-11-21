// File: v5_pipeline.cu
// Authors: Team 8 (Thụ: V5 owner)
// Description: V5 end-to-end CUDA pipeline: GPU data residency.
// External sources: none.
//
// V5 keeps the computation of the latest integrated version (V1 S1-S5 kernels
// and the V3 texture S6 kernel) and changes only the data path:
//   * A and B are uploaded once and reused by S1 and S6.
//   * gray/blur/Harris images, keypoints, descriptors, matches and H stay in
//     device buffers between stages.
//   * Only counts/status flags, canvas metadata, the crop rectangle and the
//     final panorama are copied device->host.
// Stage/kernel/copy timing and the end-to-end clock follow V1 exactly, and the
// copied byte counts are recorded so V5 can be compared with V1/V3.

#include "v5_pipeline.h"
#include "v3_pipeline.h"
#include "../core_headers/v5_stages.h"
#include "../core_headers/cuda_utils.cuh"
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <exception>
#include <random>
#include <utility>
#include <vector>

#if !defined(PANORAMA_WITH_CUDA) || !defined(PANORAMA_WITH_V3)
#error "V5 builds on the V3 texture S6: define PANORAMA_WITH_CUDA and PANORAMA_WITH_V3"
#endif

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

template <typename Function>
void measure(double& target, Function&& function) {
    struct StageScope {
        double* previous;
        explicit StageScope(double& field) : previous(v1_cuda::active_profile->stage) {
            v1_cuda::active_profile->stage = &field;
        }
        ~StageScope() { v1_cuda::active_profile->stage = previous; }
    } scope(target);
    function();
}

// Optional debug data is copied after the timed pipeline with raw cudaMemcpy,
// so it is not counted in h2d/d2h time or bytes (same policy as V1: debug
// output must not change the benchmark).
template <typename T>
void debug_download(std::vector<T>& host, const T* device, int count) {
    host.resize(static_cast<size_t>(count));
    if (count > 0)
        v1_cuda::check(cudaMemcpy(host.data(), device, host.size() * sizeof(T),
                                  cudaMemcpyDeviceToHost));
}

Image make_selftest_image(int width, int height) {
    Image image;
    image.w = width;
    image.h = height;
    image.c = 3;
    image.data.resize(static_cast<std::size_t>(width) * height * 3);
    std::mt19937 rng(2026u);
    for (uint8_t& value : image.data) value = static_cast<uint8_t>(rng() & 255u);
    return image;
}

} // namespace

bool run_v5(const Image& A, const Image& B, PipelineResult& out) {
    const bool collect_debug = out.collect_debug;
    out = PipelineResult{};
    out.collect_debug = collect_debug;
    const auto total_start = Clock::now();

    if (!valid_rgb_image(A) || !valid_rgb_image(B)) return false;

    try {
        V5HomographyResult homography{};
        int keypoint_count_a = 0, keypoint_count_b = 0, match_count = 0;
        // Persistent per-call device workspace. Allocation is part of
        // end-to-end time, exactly like the per-stage allocations of V1.
        const size_t pixels_a = static_cast<size_t>(A.w) * A.h;
        const size_t pixels_b = static_cast<size_t>(B.w) * B.h;
        const size_t pixels_max = pixels_a > pixels_b ? pixels_a : pixels_b;
        v1_cuda::Buffer<uint8_t> rgb_a(A.data.size()), rgb_b(B.data.size());
        v1_cuda::Buffer<float> gray_a(pixels_a), gray_b(pixels_b);
        v1_cuda::Buffer<float> blur_a(pixels_a), blur_b(pixels_b);
        v1_cuda::Buffer<float> response_a(pixels_a), response_b(pixels_b);
        // Scratch reused by A and B (their contents are dead after each call).
        v1_cuda::Buffer<float> tmp(pixels_max), ixx(pixels_max), iyy(pixels_max), ixy(pixels_max);
        v1_cuda::Buffer<float> weights(2 * params::kGaussRadius + 1);
        v1_cuda::Buffer<Keypoint> keypoints_a(params::kTopK), keypoints_b(params::kTopK);
        v1_cuda::Buffer<Descriptor> descriptors_a(params::kTopK), descriptors_b(params::kTopK);
        v1_cuda::Buffer<Match> matches(params::kTopK);
        v1_cuda::Buffer<V5HomographyResult> device_homography(1);

        v1_cuda::Profile profile(out.times);
        v1_cuda::ProfileScope profile_scope(profile);

        // S1: the only image upload of the whole pipeline.
        measure(out.times.gray, [&] {
            v1_cuda::copy(rgb_a.data, A.data.data(), A.data.size(), cudaMemcpyHostToDevice);
            v1_cuda::copy(rgb_b.data, B.data.data(), B.data.size(), cudaMemcpyHostToDevice);
            v5::rgb_to_gray(rgb_a.data, A.w, A.h, gray_a.data);
            v5::rgb_to_gray(rgb_b.data, B.w, B.h, gray_b.data);
        });
        measure(out.times.blur, [&] {
            v5::gaussian_blur(gray_a.data, A.w, A.h, tmp.data, weights.data, blur_a.data);
            v5::gaussian_blur(gray_b.data, B.w, B.h, tmp.data, weights.data, blur_b.data);
        });

        // S2.
        measure(out.times.harris, [&] {
            v5::harris_response(blur_a.data, A.w, A.h, ixx.data, iyy.data, ixy.data, response_a.data);
            v5::harris_response(blur_b.data, B.w, B.h, ixx.data, iyy.data, ixy.data, response_b.data);
        });
        measure(out.times.nms_topk, [&] {
            keypoint_count_a = v5::nms_topk(response_a.data, A.w, A.h, keypoints_a.data);
            keypoint_count_b = v5::nms_topk(response_b.data, B.w, B.h, keypoints_b.data);
        });
        out.nKpA = keypoint_count_a;
        out.nKpB = keypoint_count_b;

        // S3. An invalid keypoint makes V1 return an empty descriptor array.
        int descriptor_count_a = 0, descriptor_count_b = 0;
        measure(out.times.describe, [&] {
            if (v5::describe(blur_a.data, A.w, A.h, keypoints_a.data, keypoint_count_a, descriptors_a.data))
                descriptor_count_a = keypoint_count_a;
            if (v5::describe(blur_b.data, B.w, B.h, keypoints_b.data, keypoint_count_b, descriptors_b.data))
                descriptor_count_b = keypoint_count_b;
        });

        // S4.
        measure(out.times.match, [&] {
            match_count = v5::match(descriptors_a.data, descriptor_count_a,
                                    descriptors_b.data, descriptor_count_b, matches.data);
        });
        out.nMatches = match_count;
        if (match_count < 4) {
            out.times.end_to_end = elapsed_ms(total_start);
            return false;
        }

        // S5: H stays in device_homography for S6.
        bool homography_ok = false;
        measure(out.times.ransac, [&] {
            homography_ok = v5::ransac_homography(
                keypoints_a.data, keypoint_count_a, keypoints_b.data, keypoint_count_b,
                matches.data, match_count, device_homography.data, homography);
        });
        out.H = homography.H;
        out.nInliers = homography.inliers;
        if (!homography_ok) {
            out.times.end_to_end = elapsed_ms(total_start);
            return false;
        }

        // S6: reuses the resident A/B buffers and the device H.
        measure(out.times.warp_blend, [&] {
            v5::warp_blend(rgb_a.data, A.w, A.h, rgb_b.data, B.w, B.h,
                           device_homography.data, out.panorama);
        });
        v1_cuda::check(cudaDeviceSynchronize());
        out.times.end_to_end = elapsed_ms(total_start);

        if (!valid_rgb_image(out.panorama)) return false;

        if (collect_debug) {
            debug_download(out.keypointsA, keypoints_a.data, keypoint_count_a);
            debug_download(out.keypointsB, keypoints_b.data, keypoint_count_b);
            debug_download(out.matches, matches.data, match_count);
        }
        return true;
    } catch (const std::exception& error) {
        out.times.end_to_end = elapsed_ms(total_start);
        std::fprintf(stderr, "V5 CUDA pipeline failed: %s\n", error.what());
        out.panorama = Image{};
        return false;
    }
}

bool cuda_selftest_v5() {
    try {
        // Same synthetic pair as the V3 pipeline test: B is A shifted by 12 px.
        const Image a = make_selftest_image(193, 157);
        Image b = a;
        for (int y = 0; y < b.h; ++y)
            for (int x = 0; x < b.w; ++x)
                for (int c = 0; c < 3; ++c)
                    b.data[(static_cast<size_t>(y) * b.w + x) * 3 + c] =
                        x + 12 < a.w ? a.data[(static_cast<size_t>(y) * a.w + x + 12) * 3 + c] : 0;
        PipelineResult reference, resident;
        reference.collect_debug = resident.collect_debug = true;
        const bool ok_reference = run_v3(a, b, reference);
        const bool ok_resident = run_v5(a, b, resident);
        bool same = ok_reference && ok_resident &&
                    reference.nKpA == resident.nKpA && reference.nKpB == resident.nKpB &&
                    reference.nMatches == resident.nMatches &&
                    reference.nInliers == resident.nInliers &&
                    reference.panorama.w == resident.panorama.w &&
                    reference.panorama.h == resident.panorama.h &&
                    reference.panorama.data == resident.panorama.data;
        for (int i = 0; same && i < 9; ++i) same = reference.H.h[i] == resident.H.h[i];
        const bool fewer_bytes =
            resident.times.h2d_bytes < reference.times.h2d_bytes &&
            resident.times.d2h_bytes < reference.times.d2h_bytes;
        std::printf("V5 CUDA selftest %s (V3 vs V5: identical=%d, h2d bytes %.0f -> %.0f, "
                    "d2h bytes %.0f -> %.0f)\n",
                    same && fewer_bytes ? "PASS" : "FAIL", same ? 1 : 0,
                    reference.times.h2d_bytes, resident.times.h2d_bytes,
                    reference.times.d2h_bytes, resident.times.d2h_bytes);
        return same && fewer_bytes;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "V5 CUDA selftest failed: %s\n", error.what());
        return false;
    }
}
