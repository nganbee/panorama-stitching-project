// Integration parity + timing contract. Link the full V0/V1 pipeline sources.
#include "../src/core_headers/versions.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>

namespace {
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
Image random_image(int w, int h) {
    Image image; image.w = w; image.h = h; image.c = 3;
    image.data.resize(size_t(w) * h * 3);
    std::mt19937 rng(2026);
    for (auto& value : image.data) value = static_cast<uint8_t>(rng() & 255u);
    return image;
}
void compare(const Image& a, const Image& b) {
    PipelineResult cpu, gpu;
    cpu.collect_debug = gpu.collect_debug = true;
    require(run_v0(a, b, cpu), "V0 synthetic integration failed");
    require(run_v1(a, b, gpu), "V1 synthetic integration failed");
    require(cpu.nKpA == gpu.nKpA && cpu.nKpB == gpu.nKpB, "Pipeline keypoint count mismatch");
    require(cpu.nMatches == gpu.nMatches && cpu.nInliers == gpu.nInliers, "Pipeline match/inlier mismatch");
    for (int side = 0; side < 2; ++side) {
        const auto& c = side ? cpu.keypointsB : cpu.keypointsA;
        const auto& g = side ? gpu.keypointsB : gpu.keypointsA;
        require(c.size() == g.size(), "Pipeline keypoint buffer mismatch");
        for (size_t i = 0; i < c.size(); ++i)
            require(c[i].x == g[i].x && c[i].y == g[i].y, "Pipeline keypoint order mismatch");
    }
    for (size_t i = 0; i < cpu.matches.size(); ++i)
        require(cpu.matches[i].idxA == gpu.matches[i].idxA && cpu.matches[i].idxB == gpu.matches[i].idxB &&
                cpu.matches[i].dist == gpu.matches[i].dist, "Pipeline matches differ");
    for (int i = 0; i < 9; ++i) require(std::fabs(cpu.H.h[i] - gpu.H.h[i]) < 1e-5, "Pipeline H differs");
    require(cpu.panorama.w == gpu.panorama.w && cpu.panorama.h == gpu.panorama.h,
            "Pipeline canvas/crop differs");
    for (size_t i = 0; i < cpu.panorama.data.size(); ++i)
        require(std::abs(int(cpu.panorama.data[i]) - int(gpu.panorama.data[i])) <= 1,
                "Pipeline panorama pixel differs by more than one level");
    const StageTimes& t = gpu.times;
    const double stages = t.gray+t.blur+t.harris+t.nms_topk+t.describe+t.match+t.ransac+t.warp_blend;
    require(t.kernel_total > 0 && t.h2d > 0 && t.d2h > 0 && t.end_to_end > 0,
            "GPU timing categories were not populated");
    require(std::fabs(stages - t.kernel_total) < 1e-5 * std::max(1.0, stages),
            "Stage kernel times must sum to kernel_total");
    std::printf("PASS: full V1 %dx%d, matches=%d inliers=%d, kernel=%.3f H2D=%.3f D2H=%.3f E2E=%.3f ms\n",
                a.w, a.h, gpu.nMatches, gpu.nInliers, t.kernel_total,t.h2d,t.d2h,t.end_to_end);
}
} // namespace

int main() {
    try {
        // Identical pair isolates integration; a translated pair exercises B->A
        // RANSAC convention and canvas preparation without interpolation noise.
        const Image a = random_image(193, 157);
        compare(a, a);
        Image b = a;
        for (int y = 0; y < b.h; ++y) for (int x = 0; x < b.w; ++x)
            for (int c = 0; c < 3; ++c)
                b.data[(size_t(y)*b.w+x)*3+c] = x + 12 < a.w ? a.data[(size_t(y)*a.w+x+12)*3+c] : 0;
        compare(a, b);
        PipelineResult empty;
        require(!run_v1(Image{}, a, empty), "Malformed input must fail");
        Image flat = a; std::fill(flat.data.begin(), flat.data.end(), 128);
        require(!run_v1(flat, flat, empty), "No-feature input must fail");
        compare(a, a); // verify profiler scope is restored after a failed run
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr,"Pipeline FAIL: %s\n",e.what()); return 1; }
}
