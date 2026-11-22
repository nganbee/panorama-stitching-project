// Authors: Team 8; implementation assisted by OpenAI Codex.
#include "v4_pipeline.h"
#include "v3_pipeline.h"
#include "../core_headers/v1_stages.h"
#include "../core_headers/v4_stages.h"
#include <cmath>
#include <cstdio>
#include <exception>
#include <stdexcept>

bool run_v4(const Image& A, const Image& B, PipelineResult& out) {
    return panorama_detail::run_v3_with_blur(A, B, out, v4_gaussian_blur, 4);
}

bool cuda_selftest_v4() {
    try {
        Gray in; in.w = 37; in.h = 19; in.data.resize(in.w * in.h);
        for (size_t i = 0; i < in.data.size(); ++i) in.data[i] = float((i * 37) % 256);
        Gray baseline, tiled;
        v1_gaussian_blur(in, baseline); v4_gaussian_blur(in, tiled);
        if (baseline.data.size() != tiled.data.size()) throw std::runtime_error("blur size differs");
        for (size_t i = 0; i < baseline.data.size(); ++i)
            if (!std::isfinite(tiled.data[i]) ||
                std::fabs(baseline.data[i] - tiled.data[i]) > 2e-4f + 2e-6f * std::fabs(baseline.data[i]))
                throw std::runtime_error("blur values differ");
        std::puts("V4 CUDA tiled blur smoke selftest PASS (run full V4 tests for acceptance)");
        return true;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "V4 CUDA selftest failed: %s\n", e.what()); return false;
    }
}
