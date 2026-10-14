// File: v0_selftest.cpp
// Authors: Team 8; maintenance assisted by OpenAI Codex.
// Description: Self-tests for V0 Stage 1 (gray + blur) and Stage 2 (Harris + NMS + top-K).
//              Synthetic inputs only; no file I/O. Output lines: [SELFTEST] <name> PASS|FAIL <detail>
// External sources: none
#include "../core_headers/common.h"
#include "../core_headers/v0_stages.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

namespace {

bool report(const char* name, bool ok, const char* fmt = "", double a = 0, double b = 0) {
    char detail[128];
    std::snprintf(detail, sizeof detail, fmt, a, b);
    std::printf("[SELFTEST] %-28s %s %s\n", name, ok ? "PASS" : "FAIL", detail);
    return ok;
}

Image make_rgb(int w, int h, uint8_t (*f)(int, int)) {
    Image im; im.w = w; im.h = h; im.c = 3;
    im.data.resize((size_t)w * h * 3);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            uint8_t v = f(x, y);
            size_t i = ((size_t)y * w + x) * 3;
            im.data[i] = im.data[i + 1] = im.data[i + 2] = v;
        }
    return im;
}

uint8_t f_const(int, int) { return 128; }
uint8_t f_checker(int x, int y) { return (((x / 16) + (y / 16)) & 1) ? 255 : 0; }

// Gaussian kernel duplicated here on purpose: the test must not share code with the code under test.
void gauss_kernel(float* k, float sigma, int r) {
    float s = 0;
    for (int i = -r; i <= r; ++i) { k[i + r] = std::exp(-(float)(i * i) / (2.f * sigma * sigma)); s += k[i + r]; }
    for (int i = 0; i < 2 * r + 1; ++i) k[i] /= s;
}
int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

// T1: constant image -> gray and blur unchanged, no keypoints.
bool test_constant() {
    Image im = make_rgb(96, 64, f_const);
    Gray g, b, R; std::vector<Keypoint> kps;
    v0_rgb_to_gray(im, g); v0_gaussian_blur(g, b); v0_harris_response(b, R); v0_nms_topk(R, kps);
    double err = 0;
    for (float v : b.data) err = std::max(err, (double)std::fabs(v - 128.f));
    return report("constant image", err < 1e-3 && kps.empty(), "max_err=%.2e", err);
}

// T2: impulse -> symmetric Gaussian, peak at center, energy preserved (impulse far from border).
bool test_impulse() {
    const int N = 41, c = 20;
    Gray g; g.w = g.h = N; g.data.assign((size_t)N * N, 0.f);
    g.data[(size_t)c * N + c] = 1000.f;
    Gray b; v0_gaussian_blur(g, b);
    double sum = 0, asym = 0; float pk = 0; int pk_i = 0;
    for (int i = 0; i < N * N; ++i) { sum += b.data[i]; if (b.data[i] > pk) { pk = b.data[i]; pk_i = i; } }
    for (int d = 1; d <= params::kGaussRadius; ++d) {
        asym = std::max(asym, (double)std::fabs(b.data[c * N + c + d] - b.data[c * N + c - d]));
        asym = std::max(asym, (double)std::fabs(b.data[(c + d) * N + c] - b.data[(c - d) * N + c]));
    }
    bool ok = std::fabs(sum - 1000.0) < 1e-2 && asym < 1e-3 && pk_i == c * N + c;
    return report("impulse (sum, symmetry)", ok, "sum=%.3f asym=%.2e", sum, asym);
}

// T3: separable blur == direct 2D blur (same kernel, clamp border) on pseudo-random image.
bool test_separable_vs_2d() {
    const int W = 64, H = 48, r = params::kGaussRadius;
    Gray g; g.w = W; g.h = H; g.data.resize((size_t)W * H);
    std::srand(7);
    for (auto& v : g.data) v = (float)(std::rand() % 256);
    Gray b; v0_gaussian_blur(g, b);
    float k[2 * params::kGaussRadius + 1]; gauss_kernel(k, params::kGaussSigma, r);
    double maxerr = 0;
    for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
        double acc = 0;
        for (int dy = -r; dy <= r; ++dy) for (int dx = -r; dx <= r; ++dx)
            acc += (double)k[dy + r] * k[dx + r] * g.data[(size_t)clampi(y + dy, 0, H - 1) * W + clampi(x + dx, 0, W - 1)];
        maxerr = std::max(maxerr, std::fabs(acc - b.data[(size_t)y * W + x]));
    }
    return report("separable == 2D blur", maxerr < 1e-3, "max_err=%.2e", maxerr);
}

// T4: checkerboard -> every keypoint lies near a grid corner (multiples of 16), and every
// interior grid corner (16..128) has at least one keypoint within 3 px.
// Note: an ideal symmetric corner gives a plateau of equal responses; the 5x5 NMS keeps plateau
// points that are >= 3 px apart, so several keypoints per corner are expected.
bool test_checkerboard() {
    Image im = make_rgb(160, 160, f_checker);
    Gray g, b, R; std::vector<Keypoint> kps;
    v0_rgb_to_gray(im, g); v0_gaussian_blur(g, b); v0_harris_response(b, R); v0_nms_topk(R, kps);
    double worst = 0;
    for (const auto& kp : kps) {
        double dx = std::fabs(kp.x - 16.0 * std::round(kp.x / 16.0));
        double dy = std::fabs(kp.y - 16.0 * std::round(kp.y / 16.0));
        worst = std::max(worst, std::max(dx, dy));
    }
    int missed = 0;
    for (int cy = 16; cy <= 128; cy += 16)
        for (int cx = 16; cx <= 128; cx += 16) {
            bool found = false;
            for (const auto& kp : kps)
                if (std::fabs(kp.x - cx) <= 3.0 && std::fabs(kp.y - cy) <= 3.0) { found = true; break; }
            if (!found) ++missed;
        }
    bool ok = worst <= 3.0 && missed == 0;
    return report("checkerboard corners", ok, "worst_dist=%.1f missed_corners=%.0f", worst, (double)missed);
}

// T5: determinism -> two runs give identical keypoints, sorted by (score desc, y, x).
bool test_determinism() {
    Image im = make_rgb(160, 160, f_checker);
    Gray g, b, R; std::vector<Keypoint> k1, k2;
    v0_rgb_to_gray(im, g); v0_gaussian_blur(g, b); v0_harris_response(b, R); v0_nms_topk(R, k1);
    v0_rgb_to_gray(im, g); v0_gaussian_blur(g, b); v0_harris_response(b, R); v0_nms_topk(R, k2);
    bool same = k1.size() == k2.size();
    for (size_t i = 0; same && i < k1.size(); ++i)
        same = k1[i].x == k2[i].x && k1[i].y == k2[i].y && k1[i].score == k2[i].score;
    bool sorted = true;
    for (size_t i = 1; i < k1.size(); ++i) sorted = sorted && k1[i - 1].score >= k1[i].score;
    return report("determinism + ordering", same && sorted, "n=%.0f", (double)k1.size());
}

}  // namespace

bool v0_selftest_stage12() {
    bool ok = true;
    ok &= test_constant();
    ok &= test_impulse();
    ok &= test_separable_vs_2d();
    ok &= test_checkerboard();
    ok &= test_determinism();
    std::printf("[SELFTEST] stage1-2 overall: %s\n", ok ? "ALL PASS" : "SOME FAILED");
    return ok;
}
