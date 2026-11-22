// Authors: Team 8; V2 coverage assisted by OpenAI Codex.
// External sources: none.
// Integration parity + timing contract. Link the full V0/V1 pipeline sources.
#include "../../src/core_headers/versions.h"
#include "../../src/core_headers/v0_stages.h"
#include "../../src/core_headers/v1_stages.h"
#include "verify_helpers.h"
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
void compare(const Image& a, const Image& b, int version) {
    PipelineResult cpu, gpu;
    cpu.collect_debug = gpu.collect_debug = true;
    require(run_v0(a, b, cpu), "V0 synthetic integration failed");
    require(get_version(version)(a, b, gpu), "CUDA synthetic integration failed");
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
    require(v2_projection_error(cpu.H, gpu.H, b.w, b.h, cpu.keypointsB) <= 1e-3,
            "Pipeline H projection error exceeds 1e-3 px");
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
    std::printf("PASS: full V%d %dx%d, matches=%d inliers=%d, kernel=%.3f H2D=%.3f D2H=%.3f E2E=%.3f ms\n",
                version, a.w, a.h, gpu.nMatches, gpu.nInliers, t.kernel_total,t.h2d,t.d2h,t.end_to_end);
}
} // namespace

int main() {
    try {
        // Identical pair isolates integration; a translated pair exercises B->A
        // RANSAC convention and canvas preparation without interpolation noise.
        const Image a = random_image(193, 157);
        Image b = a;
        for (int y = 0; y < b.h; ++y) for (int x = 0; x < b.w; ++x)
            for (int c = 0; c < 3; ++c)
                b.data[(size_t(y)*b.w+x)*3+c] = x + 12 < a.w ? a.data[(size_t(y)*a.w+x+12)*3+c] : 0;
        Image flat = a; std::fill(flat.data.begin(), flat.data.end(), 128);
        for (int version : {1, 2}) {
            compare(a, a, version);
            compare(a, b, version);
            PipelineResult empty;
            require(!get_version(version)(Image{}, a, empty), "Malformed input must fail");
            require(!get_version(version)(flat, flat, empty), "No-feature input must fail");
            compare(a, a, version); // profiler restored after a failed run
        }
        PipelineResult v1, v2;
        v1.collect_debug = v2.collect_debug = true;
        require(run_v1(a,b,v1) && run_v2(a,b,v2), "Direct V1/V2 integration failed");
        require(v2_equal_matches(v1.matches,v2.matches), "V1/V2 ordered matches differ");
        require(v2_projection_error(v1.H,v2.H,b.w,b.h,v1.keypointsB)==0, "V1/V2 H projection differs");
        require(v1.nInliers==v2.nInliers && v1.panorama.w==v2.panorama.w &&
                v1.panorama.h==v2.panorama.h && v1.panorama.data==v2.panorama.data,
                "V1/V2 inliers or panorama differ");
        std::puts("PASS: direct V1/V2 integration");
        // S5 is inherited: explicitly preserve failure behavior for too few
        // valid correspondences and degenerate geometry in the V2 test suite.
        std::vector<Keypoint> line{{0,0,0},{1,2,0},{2,4,0},{3,6,0}};
        for (int n : {3,4}) {
            std::vector<Match> bad;
            for (int i=0; i<n; ++i) bad.push_back({i,i,0});
            Homography h{}; int inliers=0;
            require(!v0_ransac_homography(line,line,bad,h,inliers), "V0 should reject invalid S5 geometry");
            require(!v1_ransac_homography(line,line,bad,h,inliers), "Inherited GPU S5 should reject invalid geometry");
        }
        std::puts("PASS: insufficient/degenerate correspondences rejected");
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr,"Pipeline FAIL: %s\n",e.what()); return 1; }
}
