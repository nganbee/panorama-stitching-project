// File: cuda_pipeline.cu
// Authors: Team 8
// Description: Version 1 end-to-end CUDA pipeline.
//
// Basic GPU S1-S6. Stage/kernel/copy times use CUDA events; end-to-end uses
// the host clock and includes allocations, launches, synchronization and copies.
// Intermediate host buffers are allowed in V1; device residency belongs to V5.

#include "cuda_pipeline.h"
#include "../core_headers/v1_stages.h"
#include "../core_headers/cuda_utils.cuh"
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <random>
#include <string>
#include <utility>

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

Image make_selftest_image(int width, int height) {
    Image image;
    image.w = width;
    image.h = height;
    image.c = 3;
    image.data.resize(static_cast<std::size_t>(width) * height * 3);
    std::mt19937 rng(20261111u);
    for (uint8_t& value : image.data) value = static_cast<uint8_t>(rng() & 255u);
    return image;
}

} // namespace

bool run_v1(const Image& A, const Image& B, PipelineResult& out) {
    const bool collect_debug = out.collect_debug;
    out = PipelineResult{};
    out.collect_debug = collect_debug;
    const auto total_start = Clock::now();

    if (!valid_rgb_image(A) || !valid_rgb_image(B)) return false;

    try {
        v1_cuda::Profile profile(out.times);
        v1_cuda::ProfileScope profile_scope(profile);
        Gray gray_a, gray_b, blur_a, blur_b, response_a, response_b;
        std::vector<Keypoint> keypoints_a, keypoints_b;
        std::vector<Descriptor> descriptors_a, descriptors_b;
        std::vector<Match> matches;

        // S1: each stage wrapper is synchronous and processes A and B in turn.
        measure(out.times.gray, [&] {
            v1_rgb_to_gray(A, gray_a);
            v1_rgb_to_gray(B, gray_b);
        });
        measure(out.times.blur, [&] {
            v1_gaussian_blur(gray_a, blur_a);
            v1_gaussian_blur(gray_b, blur_b);
        });

        // S2.
        measure(out.times.harris, [&] {
            v1_harris_response(blur_a, response_a);
            v1_harris_response(blur_b, response_b);
        });
        measure(out.times.nms_topk, [&] {
            v1_nms_topk(response_a, keypoints_a);
            v1_nms_topk(response_b, keypoints_b);
        });
        out.nKpA = static_cast<int>(keypoints_a.size());
        out.nKpB = static_cast<int>(keypoints_b.size());

        // S3.
        measure(out.times.describe, [&] {
            v1_describe(blur_a, keypoints_a, descriptors_a);
            v1_describe(blur_b, keypoints_b, descriptors_b);
        });

        // S4.
        measure(out.times.match, [&] {
            v1_match(descriptors_a, descriptors_b, matches);
        });
        out.nMatches = static_cast<int>(matches.size());
        if (matches.size() < 4) {
            out.times.end_to_end = elapsed_ms(total_start);
            return false;
        }

        // S5: sampling, DLT, scoring and refinement all execute on the GPU.
        bool homography_ok = false;
        measure(out.times.ransac, [&] {
            homography_ok = v1_ransac_homography(
                keypoints_a, keypoints_b, matches, out.H, out.nInliers);
        });
        if (!homography_ok) {
            out.times.end_to_end = elapsed_ms(total_start);
            return false;
        }

        // S6.
        measure(out.times.warp_blend, [&] {
            v1_warp_blend(A, B, out.H, out.panorama);
        });
        v1_cuda::check(cudaDeviceSynchronize());
        out.times.end_to_end = elapsed_ms(total_start);

        if (!valid_rgb_image(out.panorama)) return false;

        if (collect_debug) {
            out.keypointsA = std::move(keypoints_a);
            out.keypointsB = std::move(keypoints_b);
            out.matches = std::move(matches);
        }
        return true;
    } catch (const std::exception& error) {
        out.times.end_to_end = elapsed_ms(total_start);
        std::fprintf(stderr, "V1 CUDA pipeline failed: %s\n", error.what());
        out.panorama = Image{};
        return false;
    }
}

bool cuda_selftest(int version) {
    if (version != 1) {
        std::fprintf(stderr, "CUDA selftest currently supports only Version 1.\n");
        return false;
    }

    try {
        const Image image = make_selftest_image(97, 83);
        Gray gray, blur, response;
        std::vector<Keypoint> keypoints;
        v1_rgb_to_gray(image, gray);
        v1_gaussian_blur(gray, blur);
        v1_harris_response(blur, response);
        v1_nms_topk(response, keypoints);
        const bool valid = valid_gray_image(gray) && valid_gray_image(blur) &&
                           valid_gray_image(response);
        std::printf("V1 CUDA S1-S2 smoke selftest %s (keypoints=%zu)\n",
                    valid ? "PASS" : "FAIL", keypoints.size());
        return valid;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "V1 CUDA selftest failed: %s\n", error.what());
        return false;
    }
}
