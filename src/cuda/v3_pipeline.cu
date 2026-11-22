// File: v3_pipeline.cu
// Authors: Team 8
// Description: V2 S1–S5 with V3 texture warp and compressed crop stack.

#include "v3_pipeline.h"
#include "v2_pipeline.h"
#include "../core_headers/v1_stages.h"
#include "../core_headers/v3_stages.h"
#include <cstdio>
#include <exception>
#include <random>

bool run_v3(const Image& A, const Image& B, PipelineResult& out) {
    return pipeline_detail::run_v2_with_stage6(A, B, out, v3_warp_blend, "V3");
}

bool cuda_selftest_v3() {
    try {
        Image image;
        image.w = 97;
        image.h = 83;
        image.c = 3;
        image.data.resize(static_cast<std::size_t>(image.w) * image.h * image.c);
        std::mt19937 rng(20261111u);
        for (uint8_t& value : image.data) value = static_cast<uint8_t>(rng() & 255u);

        const Homography identity{{1,0,0,0,1,0,0,0,1}};
        Image baseline, texture;
        v1_warp_blend(image, image, identity, baseline);
        v3_warp_blend(image, image, identity, texture);
        const bool valid = valid_rgb_image(texture) && baseline.w == texture.w &&
                           baseline.h == texture.h && baseline.data == texture.data;
        std::printf("V3 CUDA S6 smoke selftest %s\n", valid ? "PASS" : "FAIL");
        return valid;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "V3 CUDA selftest failed: %s\n", error.what());
        return false;
    }
}
