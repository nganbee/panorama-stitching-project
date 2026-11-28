// V6 pipeline parity vs V0/V5; synthetic pairs and error cases.
// Authors: Nguyen Duy Khanh; assisted by OpenAI Codex.
#include "../../core_headers/versions.h"
#include "../../libraries/stb_image.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>

#if !defined(PANORAMA_WITH_V6) || !defined(PANORAMA_WITH_V5)
#error "Build with -DPANORAMA_WITH_CUDA -DPANORAMA_WITH_V2 -DPANORAMA_WITH_V3 -DPANORAMA_WITH_V4 -DPANORAMA_WITH_V5 -DPANORAMA_WITH_V6"
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
// V6 changes only S4 work distribution, so it must equal V5 exactly.
void compare_exact(const PipelineResult& v4, const PipelineResult& v5) {
    require(v4.nKpA == v5.nKpA && v4.nKpB == v5.nKpB && v4.nMatches == v5.nMatches &&
            v4.nInliers == v5.nInliers, "V5/V6 counts differ");
    for (int side = 0; side < 2; ++side) {
        const auto& a = side ? v4.keypointsB : v4.keypointsA;
        const auto& g = side ? v5.keypointsB : v5.keypointsA;
        require(a.size() == g.size(), "V5/V6 keypoint buffers differ");
        for (size_t i = 0; i < a.size(); ++i)
            require(a[i].x == g[i].x && a[i].y == g[i].y && a[i].score == g[i].score,
                    "V5/V6 keypoints differ");
    }
    require(v4.matches.size() == v5.matches.size(), "V5/V6 match buffers differ");
    for (size_t i = 0; i < v4.matches.size(); ++i)
        require(v4.matches[i].idxA == v5.matches[i].idxA && v4.matches[i].idxB == v5.matches[i].idxB &&
                v4.matches[i].dist == v5.matches[i].dist, "V5/V6 matches differ");
    for (int i = 0; i < 9; ++i) require(v4.H.h[i] == v5.H.h[i], "V5/V6 homographies differ");
    require(v4.panorama.w == v5.panorama.w && v4.panorama.h == v5.panorama.h &&
            v4.panorama.data == v5.panorama.data, "V5/V6 panorama bytes differ");
}
void check_timing(const StageTimes& t) {
    const double stages = t.gray + t.blur + t.harris + t.nms_topk + t.describe + t.match + t.ransac + t.warp_blend;
    require(t.warp_blend > 0 && t.kernel_total > 0 && t.h2d > 0 && t.d2h > 0 && t.end_to_end > 0,
            "V6 timing categories missing");
    require(std::fabs(stages - t.kernel_total) <= 1e-5 * std::max(1.0, stages), "V6 kernel timing sum differs");
}
void check_pair(const Image& a, const Image& b) {
    PipelineResult cpu, baseline, target;
    cpu.collect_debug = baseline.collect_debug = target.collect_debug = true;
    require(run_v0(a,b,cpu), "V0 failed; panorama parity NOT verified");
    require(run_v5(a,b,baseline), "V5 failed");
    require(run_v6(a,b,target), "V6 failed");
    compare_plan(cpu,target,b);
    compare_exact(baseline,target);
    check_timing(target.times);
    require(baseline.times.h2d_bytes == target.times.h2d_bytes &&
            baseline.times.d2h_bytes == target.times.d2h_bytes, "V6 changed residency/copy bytes");
    std::puts("PASS: V6 pipeline matches V0/V5; residency unchanged");
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
        require(!run_v6(Image{}, a, output), "Malformed input accepted");
        Image flat = a; std::fill(flat.data.begin(), flat.data.end(), 128);
        require(!run_v6(flat, flat, output), "No-feature input accepted");
        check_pair(a, a); // the device stays usable after expected failures
        std::puts("PASS: V6 expected error cases");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "V6 pipeline FAIL: %s\n", error.what()); return 1;
    }
}
