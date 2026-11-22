// Authors: Team 8; implementation assisted by OpenAI Codex.
// Isolated blur parity, halo/partial blocks, and propagated S2/S3/S4 parity.
#include "../../core_headers/v0_stages.h"
#include "../../core_headers/v1_stages.h"
#include "../../core_headers/v2_stages.h"
#include "../../core_headers/v4_stages.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>

namespace {
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void compare_blur(const Gray& a, const Gray& b) {
    require(a.w == b.w && a.h == b.h && a.data.size() == b.data.size(), "Blur shape differs");
    for (size_t i = 0; i < a.data.size(); ++i)
        require(std::isfinite(b.data[i]) &&
                std::fabs(a.data[i] - b.data[i]) <= 2e-4f + 2e-6f * std::fabs(a.data[i]),
                "Blur error exceeds V0 tolerance");
}
void compare_keypoints(const std::vector<Keypoint>& a, const std::vector<Keypoint>& b) {
    require(a.size() == b.size(), "Keypoint count differs");
    for (size_t i = 0; i < a.size(); ++i)
        require(a[i].x == b[i].x && a[i].y == b[i].y && std::isfinite(b[i].score) &&
                std::fabs(a[i].score - b[i].score) <= 5e-2f + 2e-5f * std::fabs(a[i].score),
                "Keypoint coordinates/order/scores differ");
}
void propagated_parity() {
    Gray in; in.w = 193; in.h = 157; in.data.resize(size_t(in.w) * in.h);
    std::mt19937 rng(2026);
    for (float& x : in.data) x = float(rng() & 255u);
    Gray blur[3], response[3];
    std::vector<Keypoint> kp[3]; std::vector<Descriptor> desc[3];
    v0_gaussian_blur(in, blur[0]); v1_gaussian_blur(in, blur[1]); v4_gaussian_blur(in, blur[2]);
    v0_harris_response(blur[0], response[0]); v0_nms_topk(response[0], kp[0]);
    v0_describe(blur[0], kp[0], desc[0]);
    for (int i = 1; i < 3; ++i) {
        v1_harris_response(blur[i], response[i]); v1_nms_topk(response[i], kp[i]);
        v1_describe(blur[i], kp[i], desc[i]);
        compare_blur(blur[0], blur[i]); compare_keypoints(kp[0], kp[i]);
        require(desc[0].size() == desc[i].size(), "Descriptor count differs");
        for (size_t j = 0; j < desc[0].size(); ++j)
            for (int word = 0; word < 4; ++word)
                require(desc[0][j].bits[word] == desc[i][j].bits[word], "BRIEF bit differs");
    }
    // Cross-version descriptors should produce the same ordered matching output.
    std::vector<Match> ref, actual;
    v0_match(desc[0], desc[0], ref); v2_match(desc[1], desc[2], actual);
    require(ref.size() == actual.size(), "Match count differs");
    for (size_t i = 0; i < ref.size(); ++i)
        require(ref[i].idxA == actual[i].idxA && ref[i].idxB == actual[i].idxB &&
                ref[i].dist == actual[i].dist, "Ordered matches differ");
}
} // namespace

int main() {
    try {
        std::mt19937 rng(741);
        const int shapes[][2] = {{1,1},{1,37},{39,1},{2,3},{3,2},{7,5},
            {15,15},{16,16},{17,17},{31,33},{32,32},{33,31},{127,93},{193,157}};
        for (const auto& s : shapes) for (int pattern = 0; pattern < 4; ++pattern) {
            Gray in; in.w = s[0]; in.h = s[1]; in.data.resize(size_t(in.w) * in.h);
            for (size_t i = 0; i < in.data.size(); ++i) {
                if (pattern == 0) in.data[i] = 128.0f;
                else if (pattern == 1) in.data[i] = i == 0 ? 255.0f : 0.0f; // corner impulse
                else if (pattern == 2) in.data[i] = float(rng() & 255u);
                else in.data[i] = ((i % in.w + i / in.w) % 2) ? 255.0f : 0.0f;
            }
            Gray cpu, baseline, tiled;
            v0_gaussian_blur(in, cpu); v1_gaussian_blur(in, baseline); v4_gaussian_blur(in, tiled);
            compare_blur(cpu, tiled); compare_blur(baseline, tiled);
            // Same coefficients and arithmetic order should also preserve GPU bytes.
            require(baseline.data == tiled.data, "V3/V4 blur is not bit-identical");
        }
        Gray invalid, out; out.w = out.h = 1; out.data = {42};
        v4_gaussian_blur(invalid, out); require(out.data.empty(), "Empty input accepted");
        invalid.w = invalid.h = 2; invalid.data = {1};
        v4_gaussian_blur(invalid, out); require(out.data.empty(), "Malformed input accepted");
        propagated_parity();
        std::puts("PASS: V4 blur vs V0/V3, tiny/partial/halo inputs and S2/S3/S4 propagation");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "V4 blur FAIL: %s\n", e.what()); return 1;
    }
}
