// Authors: Team 8; implementation assisted by OpenAI Codex.
// Compare both crop paths on the shared warp output.
#include "../../core_headers/v0_stages.h"
#include "../../core_headers/v1_stages.h"
#include "../../core_headers/v3_stages.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <stdexcept>

bool stage6_crop_rects_match(const Image&, const Image&, const Homography&);

namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
Image pattern(int w, int h, unsigned seed) {
    Image image;
    image.w = w; image.h = h;
    image.data.resize(static_cast<size_t>(w) * h * 3);
    std::mt19937 rng(seed);
    for (auto& value : image.data) value = static_cast<uint8_t>(rng() & 255u);
    return image;
}
Homography translation(double x = 0, double y = 0) {
    return Homography{{1,0,x,0,1,y,0,0,1}};
}

void compare(const Image& a, const Image& b, const Homography& h, const char* name) {
    Image cpu, baseline, optimized, repeated;
    v0_warp_blend(a,b,h,cpu);
    v1_warp_blend(a,b,h,baseline);
    v3_warp_blend(a,b,h,optimized);
    v3_warp_blend(a,b,h,repeated);
    require(valid_rgb_image(cpu) && valid_rgb_image(baseline) && valid_rgb_image(optimized),
            "Positive panorama case failed; parity is NOT verified");
    require(cpu.w == optimized.w && cpu.h == optimized.h && cpu.data.size() == optimized.data.size(),
            "V0/V3 crop dimensions differ");
    double mae = 0;
    for (size_t i=0; i<cpu.data.size(); ++i) {
        const int error = std::abs(int(cpu.data[i])-int(optimized.data[i]));
        require(error <= 1, "V0/V3 RGB error exceeds 1");
        mae += error;
    }
    require(baseline.w == optimized.w && baseline.h == optimized.h && baseline.data == optimized.data,
            "V1/V3 crop or RGB bytes differ");
    require(optimized.w == repeated.w && optimized.h == repeated.h && optimized.data == repeated.data,
            "V3 is nondeterministic");
    require(stage6_crop_rects_match(a,b,h), "Crop rectangles differ on the same V1/V2 warp mask");
    std::printf("PASS: %s; crop coordinates and RGB exact vs V1; V0 RGB MAE=%.9f\n",
                name, mae/cpu.data.size());
}

void reject(const Image& a, const Image& b, const Homography& h, const char* name) {
    Image baseline, optimized;
    v1_warp_blend(a,b,h,baseline); v3_warp_blend(a,b,h,optimized);
    require(baseline.data.empty() && optimized.data.empty(), "Invalid S6 input accepted");
    std::printf("PASS: expected rejection (%s); no panorama correctness claim\n", name);
}
} // namespace

int main() {
    try {
        const Image a = pattern(37,29,1), b = pattern(31,23,2);
        compare(a,a,translation(),"identity random RGB");
        compare(a,b,translation(9,4),"integer translation");
        compare(a,b,translation(-7,-3),"negative translation");
        compare(a,b,translation(2.5,3.25),"fractional translation / manual bilinear");
        const Homography projective{{1,0.05,4,0.02,1,-2,0.0005,0.0002,1}};
        compare(a,b,projective,"perspective / feather / partial blocks");
        compare(pattern(1,1,3),pattern(1,1,4),translation(),"single pixel");
        compare(pattern(1,513,3),pattern(1,513,4),translation(),"single column");
        compare(pattern(513,1,3),pattern(513,1,4),translation(),"single row");
        const Image large = pattern(257,263,5);
        compare(large,large,translation(63.5,-7.25),"multiple crop reduction groups");
        compare(a,pattern(a.w,a.h,6),translation(90,0),"disjoint equal-area regions / crop tie ordering");
        Homography singular = translation(); singular.h[0] = singular.h[4] = 0;
        reject(a,b,singular,"singular H");
        Homography nonfinite = translation(); nonfinite.h[2] = std::numeric_limits<double>::infinity();
        reject(a,b,nonfinite,"nonfinite H");
        reject(Image{},b,translation(),"empty image");
        Image malformed = a; malformed.data.pop_back();
        reject(malformed,b,translation(),"malformed RGB buffer");
        compare(a,b,translation(),"repeat after failure");
        std::puts("PASS: V3 S6 shared warp and compressed crop stack verification");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr,"V3 S6 FAIL: %s\n",error.what());
        return 1;
    }
}
