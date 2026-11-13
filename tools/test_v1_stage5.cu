// CUDA: nvcc -O2 -std=c++17 --fmad=false tools/test_v1_stage5.cu
//       src/stage_implementations/v0_stage5.cpp -o build/stage5_cuda_selftest
// Portable helpers only: g++ -O2 -std=c++17 -x c++ -DPANORAMA_STAGE5_HOST_TEST
//       tools/test_v1_stage5.cu src/stage_implementations/v0_stage5.cpp -o build/stage5_math_test
// The production implementation is included to test its private helpers without
// introducing stage-specific algorithm headers or exported test APIs.
#include "../src/stage_implementations/v1_stage5.cu"
#include "../src/core_headers/v0_stages.h"
#include <algorithm>
#include <array>
#include <cstdio>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }

void test_samples() {
    stage5::Mt19937 actual_rng(params::kRansacSeed);
    std::mt19937 expected_rng(params::kRansacSeed);
    for (int i = 0; i < 20000; ++i)
        require(actual_rng.next() == expected_rng(), "MT19937 sequence differs from V0");
    for (int n : {4, 5, 7, 48, 60, 3000, 65537, 1073741825}) {
        std::vector<int> expected(4 * params::kRansacIters), actual(expected.size());
        std::mt19937 rng(params::kRansacSeed);
        std::uniform_int_distribution<int> pick(0, n - 1);
        for (int i = 0; i < params::kRansacIters; ++i) {
            int* sample = expected.data() + 4 * i;
            for (int size = 0; size < 4;) {
                const int value = pick(rng);
                bool duplicate = false;
                for (int j = 0; j < size; ++j) duplicate = duplicate || sample[j] == value;
                if (!duplicate) sample[size++] = value;
            }
        }
#ifdef PANORAMA_STAGE5_HOST_TEST
        stage5::Mt19937 test_rng(params::kRansacSeed);
        for (int i = 0; i < params::kRansacIters; ++i)
            stage5::sample_four(test_rng, n, actual.data() + 4 * i);
#else
        v1_cuda::Buffer<int> count(1), samples(actual.size());
        v1_cuda::copy(count.data, &n, sizeof(n), cudaMemcpyHostToDevice);
        v1_cuda::launch([&] { generate_samples_kernel<<<1, 1>>>(count.data, samples.data); });
        v1_cuda::copy(actual.data(), samples.data, actual.size() * sizeof(int), cudaMemcpyDeviceToHost);
#endif
        require(actual == expected, "RANSAC sample sequence differs from V0/toolchain");
    }
}

#ifdef PANORAMA_STAGE5_HOST_TEST
// Test adapter only: exercise the actual fixed-buffer numerical helpers against
// the independent V0 oracle. This never exists in a production CUDA binary.
bool run_test_stage5(const std::vector<Keypoint>& a, const std::vector<Keypoint>& b,
                     const std::vector<Match>& matches, Homography& H, int& inliers) {
    H = Homography{}; inliers = 0;
    std::vector<stage5::Point> src, dst;
    for (const Match& m : matches) {
        if (m.idxA < 0 || size_t(m.idxA) >= a.size() || m.idxB < 0 || size_t(m.idxB) >= b.size()) continue;
        if (!stage5::finite(a[m.idxA].x) || !stage5::finite(a[m.idxA].y) ||
            !stage5::finite(b[m.idxB].x) || !stage5::finite(b[m.idxB].y)) continue;
        src.push_back({b[m.idxB].x, b[m.idxB].y}); dst.push_back({a[m.idxA].x, a[m.idxA].y});
    }
    if (src.size() < 4) return false;
    stage5::Mt19937 rng(params::kRansacSeed);
    std::vector<int> best, selected;
    double best_error = HUGE_VAL;
    for (int i = 0; i < params::kRansacIters; ++i) {
        int sample[4]; stage5::sample_four(rng, static_cast<int>(src.size()), sample);
        Homography candidate{};
        if (!stage5::estimate(src.data(), dst.data(), sample, 4, candidate.h)) continue;
        selected.clear(); double total = 0;
        for (size_t j = 0; j < src.size(); ++j) {
            const double error = stage5::error(candidate.h, src[j], dst[j]);
            if (error <= 9.0) { selected.push_back(static_cast<int>(j)); total += error; }
        }
        if (selected.size() > best.size() || (selected.size() == best.size() && total < best_error)) {
            best = selected; best_error = total;
        }
    }
    if (!stage5::estimate(src.data(), dst.data(), best.data(), static_cast<int>(best.size()), H.h)) return false;
    for (size_t i = 0; i < src.size(); ++i) if (stage5::error(H.h, src[i], dst[i]) <= 9.0) ++inliers;
    return inliers >= 4;
}
#else
bool run_test_stage5(const std::vector<Keypoint>& a, const std::vector<Keypoint>& b,
                     const std::vector<Match>& matches, Homography& H, int& inliers) {
    return v1_ransac_homography(a, b, matches, H, inliers);
}
#endif

void compare(const char* name, const std::vector<Keypoint>& a, const std::vector<Keypoint>& b,
             const std::vector<Match>& matches, bool expected_ok) {
    Homography expected{}, actual{}; int ne = 0, na = 0;
    const bool ok0 = v0_ransac_homography(a, b, matches, expected, ne);
    const bool ok1 = run_test_stage5(a, b, matches, actual, na);
    require(ok0 == expected_ok && ok1 == ok0, name);
    require(ne == na, "S5 refined inlier count mismatch");
    if (ok0) {
        // Project a grid instead of comparing arbitrarily scaled matrix entries.
        for (int y = 0; y <= 600; y += 100) for (int x = 0; x <= 800; x += 100) {
            const double z0 = expected.h[6] * x + expected.h[7] * y + expected.h[8];
            const double z1 = actual.h[6] * x + actual.h[7] * y + actual.h[8];
            for (int row = 0; row < 2; ++row) {
                const double p0 = (expected.h[row * 3] * x + expected.h[row * 3 + 1] * y + expected.h[row * 3 + 2]) / z0;
                const double p1 = (actual.h[row * 3] * x + actual.h[row * 3 + 1] * y + actual.h[row * 3 + 2]) / z1;
                require(std::fabs(p0 - p1) < 1e-4, "S5 reprojection mismatch");
            }
        }
    }
    Homography repeated{}; int nr = 0;
    require(run_test_stage5(a, b, matches, repeated, nr) == ok1 && nr == na, "S5 nondeterministic status/count");
    for (int i = 0; i < 9; ++i) require(repeated.h[i] == actual.h[i], "S5 nondeterministic H");
    std::printf("PASS: %s (ok=%d, inliers=%d)\n", name, int(ok1), na);
}

void test_geometry() {
    for (bool perspective : {false, true}) {
        const Homography truth = perspective ? Homography{{1.01, .03, 37, -.02, .98, -19, .00015, -.0001, 1}}
                                             : Homography{{1, 0, 37, 0, 1, -19, 0, 0, 1}};
        std::vector<Keypoint> a, b; std::vector<Match> matches;
        for (int y = 0; y < 6; ++y) for (int x = 0; x < 8; ++x) {
            const float sx = 80.0f + 70 * x, sy = 90.0f + 65 * y;
            const double z = truth.h[6] * sx + truth.h[7] * sy + 1;
            b.push_back({sx, sy, 1});
            a.push_back({float((truth.h[0]*sx + truth.h[1]*sy + truth.h[2])/z),
                         float((truth.h[3]*sx + truth.h[4]*sy + truth.h[5])/z), 1});
            const int i = static_cast<int>(a.size()) - 1;
            matches.push_back({i, i, 0});
        }
        for (int i = 0; i < 12; ++i) matches.push_back({i, 47 - i, 200});
        compare(perspective ? "perspective + outliers" : "translation + outliers", a, b, matches, true);
        matches.insert(matches.begin(), {-1, 0, 0}); matches.push_back({0, 10000, 0});
        a.push_back({std::numeric_limits<float>::quiet_NaN(), 0, 1});
        b.push_back({0, std::numeric_limits<float>::infinity(), 1});
        matches.push_back({48, 0, 0}); matches.push_back({0, 48, 0});
        compare("invalid indices and nonfinite coordinates filtered", a, b, matches, true);
    }
    const std::vector<Keypoint> corners{{0,0,1},{100,0,1},{0,100,1},{100,100,1}};
    const std::vector<Match> four{{0,0,0},{1,1,0},{2,2,0},{3,3,0}};
    compare("exactly four points", corners, corners, four, true);
    compare("fewer than four", corners, corners, {{0,0,0},{1,1,0},{2,2,0}}, false);
    compare("empty", {}, {}, {}, false);
    compare("no valid matches", corners, corners, {{-1,0,0},{99,0,0},{0,99,0},{0,-1,0}}, false);
    const std::vector<Keypoint> line{{0,0,1},{1,1,1},{2,2,1},{3,3,1}};
    compare("collinear degeneracy", line, line, four, false);
    compare("repeated point degeneracy", corners, corners, {{0,0,0},{0,0,0},{0,0,0},{0,0,0}}, false);
}
} // namespace

int main() {
    try {
        test_samples(); test_geometry();
#ifdef PANORAMA_STAGE5_HOST_TEST
        std::puts("PASS: portable S5 math/sampling only (GPU execution not tested)");
#else
        std::puts("PASS: S5 CUDA sample and V0 parity tests");
#endif
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "S5 FAIL: %s\n", e.what()); return 1; }
}
