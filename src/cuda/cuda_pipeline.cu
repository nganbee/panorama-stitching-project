// File: cuda_pipeline.cu
// Authors: Team 8
// Description: Version 1 end-to-end CUDA pipeline.
//
// V1 is deliberately a correctness-first baseline. Each current CUDA stage
// exposes a synchronous host-buffer wrapper, so a stage timing includes its
// allocations and host/device transfers. Version 2 can replace this glue with
// a persistent device workspace and separate CUDA-event timing.

#include "cuda_pipeline.h"
#include "../core_headers/v0_stages.h"
#include "../core_headers/v1_stages.h"
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
double measure(Function&& function) {
    const auto start = Clock::now();
    function();
    return elapsed_ms(start);
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
        Gray gray_a, gray_b, blur_a, blur_b, response_a, response_b;
        std::vector<Keypoint> keypoints_a, keypoints_b;
        std::vector<Descriptor> descriptors_a, descriptors_b;
        std::vector<Match> matches;

        // S1: each stage wrapper is synchronous and processes A and B in turn.
        out.times.gray = measure([&] {
            v1_rgb_to_gray(A, gray_a);
            v1_rgb_to_gray(B, gray_b);
        });
        out.times.blur = measure([&] {
            v1_gaussian_blur(gray_a, blur_a);
            v1_gaussian_blur(gray_b, blur_b);
        });

        // S2.
        out.times.harris = measure([&] {
            v1_harris_response(blur_a, response_a);
            v1_harris_response(blur_b, response_b);
        });
        out.times.nms_topk = measure([&] {
            v1_nms_topk(response_a, keypoints_a);
            v1_nms_topk(response_b, keypoints_b);
        });
        out.nKpA = static_cast<int>(keypoints_a.size());
        out.nKpB = static_cast<int>(keypoints_b.size());

        // S3.
        out.times.describe = measure([&] {
            v1_describe(blur_a, keypoints_a, descriptors_a);
            v1_describe(blur_b, keypoints_b, descriptors_b);
        });

        // S4.
        out.times.match = measure([&] {
            v1_match(descriptors_a, descriptors_b, matches);
        });
        out.nMatches = static_cast<int>(matches.size());
        if (matches.size() < 4) {
            out.times.end_to_end = elapsed_ms(total_start);
            return false;
        }

        // S5: CPU V0 reference implementation for the first integrated V1.
        bool homography_ok = false;
        out.times.ransac = measure([&] {
            homography_ok = v0_ransac_homography(
                keypoints_a, keypoints_b, matches, out.H, out.nInliers);
        });
        if (!homography_ok) {
            out.times.end_to_end = elapsed_ms(total_start);
            return false;
        }

        // S6.
        out.times.warp_blend = measure([&] {
            v1_warp_blend(A, B, out.H, out.panorama);
        });
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
