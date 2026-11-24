// Authors: Team 8; implementation assisted by Antigravity.
#include "v4_pipeline.h"
#include "v2_pipeline.h"
#include "../core_headers/v1_stages.h"
#include "../core_headers/v3_stages.h"
#include "../core_headers/v4_stages.h"
#include <cmath>
#include <cstdio>
#include <exception>
#include <random>
#include <stdexcept>

bool run_v4(const Image& A, const Image& B, PipelineResult& out) {
    return pipeline_detail::run_v2_with_stage6(A, B, out, v4_warp_blend, "V4", v1_gaussian_blur);
}

bool cuda_selftest_v4() {
    try {
        Image image;
        image.w = 97;
        image.h = 83;
        image.c = 3;
        image.data.resize(static_cast<std::size_t>(image.w) * image.h * image.c);
        std::mt19937 rng(20261111u);
        for (uint8_t& value : image.data) value = static_cast<uint8_t>(rng() & 255u);

        const Homography identity{{1, 0, 0, 0, 1, 0, 0, 0, 1}};
        Image baseline, optimized;
        v3_warp_blend(image, image, identity, baseline);
        v4_warp_blend(image, image, identity, optimized);
        if (!valid_rgb_image(optimized) || baseline.w != optimized.w || baseline.h != optimized.h ||
            baseline.data.size() != optimized.data.size()) {
            throw std::runtime_error("S6 warp output shape differs");
        }
        for (std::size_t i = 0; i < baseline.data.size(); ++i) {
            if (std::abs(int(baseline.data[i]) - int(optimized.data[i])) > 1) {
                throw std::runtime_error("S6 warp RGB error exceeds 1");
            }
        }
        std::puts("V4 CUDA S6 warp smoke selftest PASS (run full V4 tests for acceptance)");
        return true;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "V4 CUDA selftest failed: %s\n", e.what());
        return false;
    }
}
