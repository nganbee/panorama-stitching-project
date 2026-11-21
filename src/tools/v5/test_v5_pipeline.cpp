// File: test_v5_pipeline.cpp
// Authors: Team 8 (Thụ: V5 owner)
// Description: V5 (GPU data residency) correctness and transfer test.
// External sources: none.
//
// No args: synthetic pairs and error cases. Two args: a real image pair.
// V5 must reproduce V3 (its kernel baseline) bit for bit, stay within the
// PLAN tolerances of V0, and copy fewer host<->device bytes than V1/V3.
#include "../../core_headers/versions.h"
#include "../../libraries/stb_image.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>

#if !defined(PANORAMA_WITH_V5)
#error "Build with -DPANORAMA_WITH_CUDA -DPANORAMA_WITH_V3 -DPANORAMA_WITH_V5"
#endif

namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
Image load(const char* path) {
    Image image;
    int channels = 0;
    uint8_t* data = stbi_load(path, &image.w, &image.h, &channels, 3);
    if (!data) throw std::runtime_error(std::string("Cannot load ") + path);
    image.data.assign(data, data + static_cast<size_t>(image.w) * image.h * 3);
    stbi_image_free(data);
    return image;
}
Image random_image(int w, int h) {
    Image image; image.w = w; image.h = h;
    image.data.resize(static_cast<size_t>(w) * h * 3);
    std::mt19937 rng(2026);
    for (auto& value : image.data) value = static_cast<uint8_t>(rng() & 255u);
    return image;
}
void project_compare(const Homography& a, const Homography& b, double x, double y) {
    const double da = a.h[6] * x + a.h[7] * y + a.h[8], db = b.h[6] * x + b.h[7] * y + b.h[8];
    require(std::isfinite(da) && std::isfinite(db) && std::fabs(da) > 1e-12 && std::fabs(db) > 1e-12,
            "Invalid homography projection");
    const double dx = (a.h[0] * x + a.h[1] * y + a.h[2]) / da - (b.h[0] * x + b.h[1] * y + b.h[2]) / db;
    const double dy = (a.h[3] * x + a.h[4] * y + a.h[5]) / da - (b.h[3] * x + b.h[4] * y + b.h[5]) / db;
    require(std::isfinite(dx) && std::isfinite(dy) && std::hypot(dx, dy) <= 1e-3,
            "Homography projection error exceeds 1e-3 px");
}
// PLAN correctness contract against V0.
void compare_plan(const PipelineResult& ref, const PipelineResult& gpu, const Image& b) {
    require(ref.nKpA == gpu.nKpA && ref.nKpB == gpu.nKpB, "Keypoint counts differ");
    require(ref.nMatches == gpu.nMatches && ref.nInliers == gpu.nInliers, "Match/inlier counts differ");
    for (int side = 0; side < 2; ++side) {
        const auto& a = side ? ref.keypointsB : ref.keypointsA;
        const auto& g = side ? gpu.keypointsB : gpu.keypointsA;
        require(a.size() == g.size(), "Keypoint buffers differ");
        for (size_t i = 0; i < a.size(); ++i)
            require(a[i].x == g[i].x && a[i].y == g[i].y && std::isfinite(g[i].score) &&
                    std::fabs(a[i].score - g[i].score) <= 5e-2f + 2e-5f * std::fabs(a[i].score),
                    "Keypoint coordinates/order/scores differ");
    }
    require(ref.matches.size() == gpu.matches.size(), "Match buffers differ");
    for (size_t i = 0; i < ref.matches.size(); ++i)
        require(ref.matches[i].idxA == gpu.matches[i].idxA && ref.matches[i].idxB == gpu.matches[i].idxB &&
                ref.matches[i].dist == gpu.matches[i].dist, "Ordered matches differ");
    for (int y = 0; y <= 4; ++y) for (int x = 0; x <= 4; ++x)
        project_compare(ref.H, gpu.H, (b.w - 1) * x / 4.0, (b.h - 1) * y / 4.0);
    for (const auto& point : ref.keypointsB) project_compare(ref.H, gpu.H, point.x, point.y);
    require(ref.panorama.w == gpu.panorama.w && ref.panorama.h == gpu.panorama.h &&
            ref.panorama.data.size() == gpu.panorama.data.size(), "Panorama/crop dimensions differ");
    double mae = 0;
    for (size_t i = 0; i < ref.panorama.data.size(); ++i) {
        const int error = std::abs(int(ref.panorama.data[i]) - int(gpu.panorama.data[i]));
        require(error <= 1, "Panorama RGB error exceeds 1"); mae += error;
    }
    std::printf("  panorama MAE vs V0=%.9f\n", mae / ref.panorama.data.size());
}
// V5 changes only the data path, so it must equal V3 exactly.
void compare_exact(const PipelineResult& v3, const PipelineResult& v5) {
    require(v3.nKpA == v5.nKpA && v3.nKpB == v5.nKpB && v3.nMatches == v5.nMatches &&
            v3.nInliers == v5.nInliers, "V3/V5 counts differ");
    for (int side = 0; side < 2; ++side) {
        const auto& a = side ? v3.keypointsB : v3.keypointsA;
        const auto& g = side ? v5.keypointsB : v5.keypointsA;
        require(a.size() == g.size(), "V3/V5 keypoint buffers differ");
        for (size_t i = 0; i < a.size(); ++i)
            require(a[i].x == g[i].x && a[i].y == g[i].y && a[i].score == g[i].score,
                    "V3/V5 keypoints differ");
    }
    require(v3.matches.size() == v5.matches.size(), "V3/V5 match buffers differ");
    for (size_t i = 0; i < v3.matches.size(); ++i)
        require(v3.matches[i].idxA == v5.matches[i].idxA && v3.matches[i].idxB == v5.matches[i].idxB &&
                v3.matches[i].dist == v5.matches[i].dist, "V3/V5 matches differ");
    for (int i = 0; i < 9; ++i) require(v3.H.h[i] == v5.H.h[i], "V3/V5 homographies differ");
    require(v3.panorama.w == v5.panorama.w && v3.panorama.h == v5.panorama.h &&
            v3.panorama.data == v5.panorama.data, "V3/V5 panorama bytes differ");
}
void check_timing(const StageTimes& t) {
    const double stages = t.gray + t.blur + t.harris + t.nms_topk + t.describe + t.match + t.ransac + t.warp_blend;
    require(t.warp_blend > 0 && t.kernel_total > 0 && t.h2d > 0 && t.d2h > 0 && t.end_to_end > 0,
            "V5 timing categories missing");
    require(std::fabs(stages - t.kernel_total) <= 1e-5 * std::max(1.0, stages), "V5 kernel timing sum differs");
}
void check_pair(const Image& a, const Image& b) {
    PipelineResult cpu, v1, v3, v5;
    cpu.collect_debug = v1.collect_debug = v3.collect_debug = v5.collect_debug = true;
    require(run_v0(a, b, cpu), "V0 failed; panorama parity NOT verified");
    require(run_v1(a, b, v1), "V1 pipeline failed");
    require(run_v3(a, b, v3), "V3 pipeline failed");
    require(run_v5(a, b, v5), "V5 pipeline failed");
    compare_plan(cpu, v5, b);
    compare_exact(v3, v5);
    check_timing(v5.times);
    // Only the two RGB inputs go up; only metadata and the panorama come back.
    const double input_bytes = static_cast<double>(a.data.size() + b.data.size());
    require(v5.times.h2d_bytes == input_bytes, "V5 uploads more than the two input images");
    require(v5.times.h2d_bytes < v1.times.h2d_bytes && v5.times.h2d_bytes < v3.times.h2d_bytes,
            "V5 does not reduce H2D bytes");
    require(v5.times.d2h_bytes < v1.times.d2h_bytes && v5.times.d2h_bytes < v3.times.d2h_bytes,
            "V5 does not reduce D2H bytes");
    require(v5.times.d2h_bytes >= static_cast<double>(v5.panorama.data.size()),
            "V5 D2H bytes smaller than the panorama");
    std::printf("PASS: V5 vs V0/V3 (%dx%d + %dx%d, matches=%d inliers=%d)\n",
                a.w, a.h, b.w, b.h, v5.nMatches, v5.nInliers);
    std::printf("  bytes H2D V1=%.0f V3=%.0f V5=%.0f | D2H V1=%.0f V3=%.0f V5=%.0f\n",
                v1.times.h2d_bytes, v3.times.h2d_bytes, v5.times.h2d_bytes,
                v1.times.d2h_bytes, v3.times.d2h_bytes, v5.times.d2h_bytes);
    std::printf("  ms H2D V3=%.3f V5=%.3f | D2H V3=%.3f V5=%.3f | E2E V3=%.3f V5=%.3f\n",
                v3.times.h2d, v5.times.h2d, v3.times.d2h, v5.times.d2h,
                v3.times.end_to_end, v5.times.end_to_end);
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 3) { check_pair(load(argv[1]), load(argv[2])); return 0; }
        if (argc != 1) { std::fprintf(stderr, "Usage: %s [left right]\n", argv[0]); return 2; }
        const Image a = random_image(193, 157);
        check_pair(a, a);
        Image b = a;
        for (int y = 0; y < b.h; ++y) for (int x = 0; x < b.w; ++x) for (int c = 0; c < 3; ++c)
            b.data[(static_cast<size_t>(y) * b.w + x) * 3 + c] =
                x + 12 < a.w ? a.data[(static_cast<size_t>(y) * a.w + x + 12) * 3 + c] : 0;
        check_pair(a, b);
        PipelineResult output;
        require(!run_v5(Image{}, a, output), "Malformed input accepted");
        Image flat = a; std::fill(flat.data.begin(), flat.data.end(), 128);
        require(!run_v5(flat, flat, output), "No-feature input accepted");
        check_pair(a, a); // the device stays usable after expected failures
        std::puts("PASS: V5 expected error cases");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "V5 pipeline FAIL: %s\n", error.what()); return 1;
    }
}
