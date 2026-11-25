// File: test_v7_pipeline.cpp
// Authors: Team 8 (Thụ: V7 owner)
// Description: V7 (S5 one warp per RANSAC hypothesis) correctness and S5 timing test.
// External sources: none.
//
// No args: S5 unit tests on synthetic matches + synthetic pipeline pairs and
// error cases. Two args: a real image pair.
//   * S5 only: v7::ransac_homography must equal v5::ransac_homography bit for
//     bit (H, inliers, status) for match counts around the warp size, with
//     outliers, duplicate points (tie-breaks) and invalid matches, and must
//     agree with V0 (same inlier count, H projection <= 1e-3 px).
//   * Pipeline: V7 must reproduce V5 bit for bit (only S5 changed, with the
//     same arithmetic order) and stay within the PLAN tolerances of V0.
#include "../../core_headers/versions.h"
#include "../../core_headers/v0_stages.h"
#include "../../core_headers/v5_stages.h"
#include "../../core_headers/v7_stages.h"
#include "../../core_headers/cuda_utils.cuh"
#include "../../libraries/stb_image.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#if !defined(PANORAMA_WITH_V7) || !defined(PANORAMA_WITH_V5)
#error "Build with -DPANORAMA_WITH_CUDA -DPANORAMA_WITH_V2 -DPANORAMA_WITH_V3 -DPANORAMA_WITH_V4 -DPANORAMA_WITH_V5 -DPANORAMA_WITH_V7"
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

// ---------------------------------------------------------------------------
// S5 unit tests (device-resident API, same as the pipeline uses).
// ---------------------------------------------------------------------------
struct S5Case {
    std::vector<Keypoint> a, b;
    std::vector<Match> matches;
};

// B points map to A by a fixed homography; a fraction of matches are outliers,
// optionally some points are duplicated (identical hypotheses -> ties) and some
// matches are invalid (out-of-range index, removed by validation).
S5Case make_s5_case(int n, double outlier_ratio, bool duplicates, bool invalid, uint32_t seed) {
    const double H[9] = {0.98, 0.03, 35.0, -0.02, 1.01, 12.0, 1.5e-5, -1.0e-5, 1.0};
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> coord(20.0f, 1200.0f), noise(-0.6f, 0.6f);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    S5Case c;
    for (int i = 0; i < n; ++i) {
        Keypoint pb;
        if (duplicates && i % 7 == 3 && i > 0) pb = c.b[i - 1];
        else { pb.x = coord(rng); pb.y = coord(rng); }
        Keypoint pa;
        const double z = H[6] * pb.x + H[7] * pb.y + H[8];
        if (unit(rng) < outlier_ratio) { pa.x = coord(rng); pa.y = coord(rng); }
        else {
            pa.x = static_cast<float>((H[0] * pb.x + H[1] * pb.y + H[2]) / z) + noise(rng);
            pa.y = static_cast<float>((H[3] * pb.x + H[4] * pb.y + H[5]) / z) + noise(rng);
        }
        c.a.push_back(pa); c.b.push_back(pb);
        Match m; m.idxA = i; m.idxB = i; m.dist = i % 64;
        if (invalid && i % 11 == 5) m.idxB = n + 3; // rejected by validate_matches_kernel
        c.matches.push_back(m);
    }
    return c;
}

template <class T>
void upload(v1_cuda::Buffer<T>& device, const std::vector<T>& host) {
    if (!host.empty())
        v1_cuda::check(cudaMemcpy(device.data, host.data(), host.size() * sizeof(T),
                                  cudaMemcpyHostToDevice));
}

struct S5Run { V5HomographyResult result; bool ok; double ms; };

template <class Function>
S5Run run_s5(const S5Case& c, Function&& ransac, int repeats = 1) {
    const int na = static_cast<int>(c.a.size()), nb = static_cast<int>(c.b.size());
    const int n = static_cast<int>(c.matches.size());
    v1_cuda::Buffer<Keypoint> a(c.a.size()), b(c.b.size());
    v1_cuda::Buffer<Match> m(c.matches.size());
    v1_cuda::Buffer<V5HomographyResult> device_result(1);
    upload(a, c.a); upload(b, c.b); upload(m, c.matches);
    S5Run run{};
    double total = 0;
    for (int r = 0; r < repeats + 1; ++r) {   // first call is a warm-up
        const auto start = std::chrono::steady_clock::now();
        run.ok = ransac(a.data, na, b.data, nb, m.data, n, device_result.data, run.result);
        v1_cuda::check(cudaDeviceSynchronize());
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        if (r) total += ms;
    }
    run.ms = total / repeats;
    return run;
}

void test_s5() {
    struct Spec { int n; double outliers; bool duplicates, invalid; };
    const Spec specs[] = {
        {4, 0.0, false, false},   {5, 0.0, false, false},   {31, 0.2, false, false},
        {32, 0.2, false, false},  {33, 0.2, true, false},   {63, 0.3, false, true},
        {64, 0.5, true, true},    {65, 0.4, false, false},  {257, 0.6, true, false},
        {1000, 0.5, false, true}, {3000, 0.7, true, false}, {3000, 0.3, false, false},
        {40, 1.0, false, false},  // pure noise: may fail; V5 and V7 must fail the same way
    };
    uint32_t seed = 7;
    for (const Spec& spec : specs) {
        const S5Case c = make_s5_case(spec.n, spec.outliers, spec.duplicates, spec.invalid, seed++);
        const S5Run r5 = run_s5(c, v5::ransac_homography);
        const S5Run r7 = run_s5(c, v7::ransac_homography);
        require(r5.ok == r7.ok, "S5: V5/V7 status differs");
        require(r5.result.ok == r7.result.ok && r5.result.inliers == r7.result.inliers,
                "S5: V5/V7 inlier counts differ");
        for (int i = 0; i < 9; ++i)
            require(r5.result.H.h[i] == r7.result.H.h[i], "S5: V5/V7 homographies differ");
        // V0 reference (CPU). Invalid matches are filtered by the GPU code;
        // V0 receives only valid ones, which is what it would see in V0.
        std::vector<Match> valid;
        for (const Match& m : c.matches)
            if (m.idxB < static_cast<int>(c.b.size())) valid.push_back(m);
        Homography h0{}; int inliers0 = 0;
        const bool ok0 = v0_ransac_homography(c.a, c.b, valid, h0, inliers0);
        require(ok0 == r7.ok, "S5: V0/V7 status differs");
        if (ok0) {
            require(inliers0 == r7.result.inliers, "S5: V0/V7 inlier counts differ");
            for (const Keypoint& p : c.b) project_compare(h0, r7.result.H, p.x, p.y);
        }
        std::printf("  S5 n=%4d outliers=%.1f dup=%d invalid=%d: ok=%d inliers=%d (V0=%d) PASS\n",
                    spec.n, spec.outliers, spec.duplicates ? 1 : 0, spec.invalid ? 1 : 0,
                    r7.ok ? 1 : 0, r7.result.inliers, inliers0);
    }
    // S5-only timing at the kTopK match count (includes allocations, like the pipeline).
    const S5Case big = make_s5_case(params::kTopK, 0.5, false, false, 99);
    const S5Run t5 = run_s5(big, v5::ransac_homography, 10);
    const S5Run t7 = run_s5(big, v7::ransac_homography, 10);
    std::printf("PASS: S5 V7 == V5 == V0 | S5 time n=%d: V5=%.3f ms V7=%.3f ms (x%.2f)\n",
                params::kTopK, t5.ms, t7.ms, t7.ms > 0 ? t5.ms / t7.ms : 0.0);
}

// ---------------------------------------------------------------------------
// Pipeline tests.
// ---------------------------------------------------------------------------
// PLAN correctness contract against V0 (same as the V5 test).
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
// V7 changes only S5 scoring with the same arithmetic order, so it equals V5.
void compare_exact(const PipelineResult& v5, const PipelineResult& v7) {
    require(v5.nKpA == v7.nKpA && v5.nKpB == v7.nKpB && v5.nMatches == v7.nMatches &&
            v5.nInliers == v7.nInliers, "V5/V7 counts differ");
    for (int side = 0; side < 2; ++side) {
        const auto& a = side ? v5.keypointsB : v5.keypointsA;
        const auto& g = side ? v7.keypointsB : v7.keypointsA;
        require(a.size() == g.size(), "V5/V7 keypoint buffers differ");
        for (size_t i = 0; i < a.size(); ++i)
            require(a[i].x == g[i].x && a[i].y == g[i].y && a[i].score == g[i].score,
                    "V5/V7 keypoints differ");
    }
    require(v5.matches.size() == v7.matches.size(), "V5/V7 match buffers differ");
    for (size_t i = 0; i < v5.matches.size(); ++i)
        require(v5.matches[i].idxA == v7.matches[i].idxA && v5.matches[i].idxB == v7.matches[i].idxB &&
                v5.matches[i].dist == v7.matches[i].dist, "V5/V7 matches differ");
    for (int i = 0; i < 9; ++i) require(v5.H.h[i] == v7.H.h[i], "V5/V7 homographies differ");
    require(v5.panorama.w == v7.panorama.w && v5.panorama.h == v7.panorama.h &&
            v5.panorama.data == v7.panorama.data, "V5/V7 panorama bytes differ");
    // Same data path as V5: identical copy volume.
    require(v5.times.h2d_bytes == v7.times.h2d_bytes && v5.times.d2h_bytes == v7.times.d2h_bytes,
            "V5/V7 transfer bytes differ");
}
void check_timing(const StageTimes& t) {
    const double stages = t.gray + t.blur + t.harris + t.nms_topk + t.describe + t.match + t.ransac + t.warp_blend;
    require(t.ransac > 0 && t.warp_blend > 0 && t.kernel_total > 0 && t.h2d > 0 && t.d2h > 0 &&
            t.end_to_end > 0, "V7 timing categories missing");
    require(std::fabs(stages - t.kernel_total) <= 1e-5 * std::max(1.0, stages), "V7 kernel timing sum differs");
}
void check_pair(const Image& a, const Image& b) {
    PipelineResult cpu, v5, v7;
    cpu.collect_debug = v5.collect_debug = v7.collect_debug = true;
    require(run_v0(a, b, cpu), "V0 failed; panorama parity NOT verified");
    require(run_v5(a, b, v5), "V5 pipeline failed");
    require(run_v7(a, b, v7), "V7 pipeline failed");
    compare_plan(cpu, v7, b);
    compare_exact(v5, v7);
    check_timing(v7.times);
    std::printf("PASS: V7 vs V0/V5 (%dx%d + %dx%d, matches=%d inliers=%d)\n",
                a.w, a.h, b.w, b.h, v7.nMatches, v7.nInliers);
    std::printf("  ms S5 V5=%.3f V7=%.3f | kernel V5=%.3f V7=%.3f | E2E V5=%.3f V7=%.3f\n",
                v5.times.ransac, v7.times.ransac, v5.times.kernel_total, v7.times.kernel_total,
                v5.times.end_to_end, v7.times.end_to_end);
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 3) { check_pair(load(argv[1]), load(argv[2])); return 0; }
        if (argc != 1) { std::fprintf(stderr, "Usage: %s [left right]\n", argv[0]); return 2; }
        test_s5();
        const Image a = random_image(193, 157);
        check_pair(a, a);
        Image b = a;
        for (int y = 0; y < b.h; ++y) for (int x = 0; x < b.w; ++x) for (int c = 0; c < 3; ++c)
            b.data[(static_cast<size_t>(y) * b.w + x) * 3 + c] =
                x + 12 < a.w ? a.data[(static_cast<size_t>(y) * a.w + x + 12) * 3 + c] : 0;
        check_pair(a, b);
        PipelineResult output;
        require(!run_v7(Image{}, a, output), "Malformed input accepted");
        Image flat = a; std::fill(flat.data.begin(), flat.data.end(), 128);
        require(!run_v7(flat, flat, output), "No-feature input accepted");
        check_pair(a, a); // the device stays usable after expected failures
        std::puts("PASS: V7 expected error cases");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "V7 pipeline FAIL: %s\n", error.what()); return 1;
    }
}
