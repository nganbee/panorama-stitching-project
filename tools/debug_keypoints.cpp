// File: tools/debug_keypoints.cpp
// Description: Debug tool for Stage 1-2 on real images (NOT part of the submitted build).
//              Runs gray -> blur -> Harris -> NMS/top-K on both images, saves gray/blur/keypoint
//              overlays and prints keypoint counts, overlap-strip counts and per-stage timings.
//              Build: g++ -O2 -std=c++17 -I src tools/debug_keypoints.cpp src/v0_stage12.cpp src/stb_impl.cpp -o dbg
//              Run:   ./dbg left.jpg right.jpg [out_prefix]
// External sources: stb_image / stb_image_write
#include "common.h"
#include "v0_stages.h"
#include "stb_image.h"
#include "stb_image_write.h"
#include <chrono>
#include <cstdio>
#include <string>
#include <algorithm>

using Clock = std::chrono::high_resolution_clock;
static double ms(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

static bool load(const char* p, Image& im) {
    int w, h, c; uint8_t* d = stbi_load(p, &w, &h, &c, 3);
    if (!d) { std::fprintf(stderr, "cannot load %s\n", p); return false; }
    im.w = w; im.h = h; im.c = 3; im.data.assign(d, d + (size_t)w * h * 3); stbi_image_free(d); return true;
}
static void save_gray(const std::string& path, const Gray& g) {
    std::vector<uint8_t> o(g.data.size());
    for (size_t i = 0; i < o.size(); ++i) o[i] = (uint8_t)std::min(255.f, std::max(0.f, g.data[i] + 0.5f));
    stbi_write_png(path.c_str(), g.w, g.h, 1, o.data(), g.w);
}
static void save_keypoints(const std::string& path, Image im, const std::vector<Keypoint>& kps) {
    for (const auto& k : kps)
        for (int d = -3; d <= 3; ++d) {                       // red cross
            int xs[2] = {(int)k.x + d, (int)k.x}, ys[2] = {(int)k.y, (int)k.y + d};
            for (int t = 0; t < 2; ++t)
                if (xs[t] >= 0 && xs[t] < im.w && ys[t] >= 0 && ys[t] < im.h) {
                    size_t i = ((size_t)ys[t] * im.w + xs[t]) * 3; im.data[i] = 255; im.data[i + 1] = 0; im.data[i + 2] = 0;
                }
        }
    stbi_write_png(path.c_str(), im.w, im.h, 3, im.data.data(), im.w * 3);
}

static void process(const char* name, const Image& im, const std::string& prefix, bool is_left) {
    Gray gray, blur, resp; std::vector<Keypoint> kps;
    double t[4] = {0, 0, 0, 0}; const int runs = 5;
    for (int r = -1; r < runs; ++r) {                         // r = -1 is warm-up
        auto s = Clock::now(); v0_rgb_to_gray(im, gray); double a = ms(s);
        s = Clock::now(); v0_gaussian_blur(gray, blur); double b = ms(s);
        s = Clock::now(); v0_harris_response(blur, resp); double c = ms(s);
        s = Clock::now(); v0_nms_topk(resp, kps); double d = ms(s);
        if (r >= 0) { t[0] += a; t[1] += b; t[2] += c; t[3] += d; }
    }
    // overlap strip: right half of the left image / left half of the right image
    int in_strip = 0;
    for (const auto& k : kps) if (is_left ? (k.x >= im.w / 2) : (k.x < im.w / 2)) ++in_strip;
    std::printf("[%s] %dx%d keypoints=%zu in_overlap_half=%d (%.0f%%)\n", name, im.w, im.h,
                kps.size(), in_strip, kps.empty() ? 0.0 : 100.0 * in_strip / kps.size());
    std::printf("[%s] avg_ms gray=%.3f blur=%.3f harris=%.3f nms_topk=%.3f\n", name,
                t[0] / runs, t[1] / runs, t[2] / runs, t[3] / runs);
    save_gray(prefix + "_" + name + "_gray.png", gray);
    save_gray(prefix + "_" + name + "_blur.png", blur);
    save_keypoints(prefix + "_" + name + "_kp.png", im, kps);
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "Usage: %s left right [out_prefix]\n", argv[0]); return 1; }
    std::string prefix = argc > 3 ? argv[3] : "dbg";
    Image A, B;
    if (!load(argv[1], A) || !load(argv[2], B)) return 1;
    process("left", A, prefix, true);
    process("right", B, prefix, false);
    return 0;
}
