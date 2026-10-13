// File: timing_stage12.cpp
// Authors: Quoc Thinh
// Description: CPU timing harness for V0 Stages 1-2 at the three required
//              resolutions. It performs one warm-up run and averages at least
//              five timed runs. The two generated images have deterministic
//              texture so the keypoint path is exercised without disk I/O.
// External sources: none.
#include "../src/core_headers/common.h"
#include "../src/core_headers/v0_stages.h"
#include "../src/libraries/stb_image.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

using Clock = std::chrono::high_resolution_clock;

double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

uint32_t hash_pixel(int x, int y, uint32_t seed) {
    uint32_t v = seed + static_cast<uint32_t>(x) * 0x9e3779b9u;
    v ^= static_cast<uint32_t>(y) + 0x85ebca6bu + (v << 6) + (v >> 2);
    v ^= v >> 16;
    v *= 0x7feb352du;
    v ^= v >> 15;
    return v;
}

Image make_image(int w, int h, uint32_t seed) {
    Image image;
    image.w = w;
    image.h = h;
    image.c = 3;
    image.data.resize(static_cast<size_t>(w) * h * 3);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const uint32_t noise = hash_pixel(x, y, seed);
            const size_t i = (static_cast<size_t>(y) * w + x) * 3;
            image.data[i + 0] = static_cast<uint8_t>((x / 17 + y / 13 + noise) & 255u);
            image.data[i + 1] = static_cast<uint8_t>((x / 29 + y / 7 + (noise >> 8)) & 255u);
            image.data[i + 2] = static_cast<uint8_t>((x / 11 + y / 31 + (noise >> 16)) & 255u);
        }
    }
    return image;
}

bool load_rgb(const char* path, Image& image) {
    int w = 0, h = 0, channels = 0;
    unsigned char* pixels = stbi_load(path, &w, &h, &channels, 3);
    if (pixels == nullptr) return false;
    image.w = w;
    image.h = h;
    image.c = 3;
    image.data.assign(pixels, pixels + static_cast<size_t>(w) * h * 3);
    stbi_image_free(pixels);
    return true;
}

Image resize_nearest(const Image& source, int w, int h) {
    Image image;
    image.w = w;
    image.h = h;
    image.c = 3;
    image.data.resize(static_cast<size_t>(w) * h * 3);
    for (int y = 0; y < h; ++y) {
        const int sy = std::min(source.h - 1, y * source.h / h);
        for (int x = 0; x < w; ++x) {
            const int sx = std::min(source.w - 1, x * source.w / w);
            const size_t dst = (static_cast<size_t>(y) * w + x) * 3;
            const size_t src = (static_cast<size_t>(sy) * source.w + sx) * 3;
            image.data[dst + 0] = source.data[src + 0];
            image.data[dst + 1] = source.data[src + 1];
            image.data[dst + 2] = source.data[src + 2];
        }
    }
    return image;
}

struct Stage12Times {
    double gray = 0.0;
    double blur = 0.0;
    double harris = 0.0;
    double nms_topk = 0.0;
    size_t keypoints_a = 0;
    size_t keypoints_b = 0;
};

struct Stage12Workspace {
    Gray gray;
    Gray blur;
    Gray response;
    std::vector<Keypoint> keypoints;
};

void process_one(const Image& image,
                 Stage12Workspace& workspace,
                 Stage12Times& time,
                 bool warmup,
                 bool is_a) {

    auto start = Clock::now();
    v0_rgb_to_gray(image, workspace.gray);
    if (!warmup) time.gray += elapsed_ms(start);

    start = Clock::now();
    v0_gaussian_blur(workspace.gray, workspace.blur);
    if (!warmup) time.blur += elapsed_ms(start);

    start = Clock::now();
    v0_harris_response(workspace.blur, workspace.response);
    if (!warmup) time.harris += elapsed_ms(start);

    start = Clock::now();
    v0_nms_topk(workspace.response, workspace.keypoints);
    if (!warmup) time.nms_topk += elapsed_ms(start);

    if (!warmup) {
        if (is_a) time.keypoints_a = workspace.keypoints.size();
        else time.keypoints_b = workspace.keypoints.size();
    }
}

void benchmark(int w, int h, int runs, const Image* base_a, const Image* base_b) {
    const Image generated_a = make_image(w, h, 0x1234u);
    const Image generated_b = make_image(w, h, 0x5678u);
    const Image resized_a = base_a != nullptr ? resize_nearest(*base_a, w, h) : Image{};
    const Image resized_b = base_b != nullptr ? resize_nearest(*base_b, w, h) : Image{};
    const Image& A = base_a != nullptr ? resized_a : generated_a;
    const Image& B = base_b != nullptr ? resized_b : generated_b;

    Stage12Workspace workspace_a;
    Stage12Workspace workspace_b;
    Stage12Times warmup;
    process_one(A, workspace_a, warmup, true, true);
    process_one(B, workspace_b, warmup, true, false);

    Stage12Times sum;
    for (int run = 0; run < runs; ++run) {
        process_one(A, workspace_a, sum, false, true);
        process_one(B, workspace_b, sum, false, false);
    }

    const double inv_runs = 1.0 / static_cast<double>(runs);
    std::printf("resolution=%dx%d,runs=%d,avg_ms_gray=%.3f,avg_ms_blur=%.3f,"
                "avg_ms_harris=%.3f,avg_ms_nms_topk=%.3f,keypoints_A=%zu,keypoints_B=%zu\n",
                w, h, runs,
                sum.gray * inv_runs,
                sum.blur * inv_runs,
                sum.harris * inv_runs,
                sum.nms_topk * inv_runs,
                sum.keypoints_a,
                sum.keypoints_b);
}

}  // namespace

int main(int argc, char** argv) {
    const int runs = argc >= 2 ? std::atoi(argv[1]) : 5;
    if (runs < 5) {
        std::fprintf(stderr, "Usage: %s [runs] [left_image right_image], where runs must be at least 5.\n", argv[0]);
        return 2;
    }

    Image base_a, base_b;
    const Image* base_a_ptr = nullptr;
    const Image* base_b_ptr = nullptr;
    if (argc == 4) {
        if (!load_rgb(argv[2], base_a) || !load_rgb(argv[3], base_b)) {
            std::fprintf(stderr, "Cannot load timing input images.\n");
            return 1;
        }
        base_a_ptr = &base_a;
        base_b_ptr = &base_b;
    } else if (argc != 1 && argc != 2) {
        std::fprintf(stderr, "Usage: %s [runs] [left_image right_image]\n", argv[0]);
        return 2;
    }

    benchmark(1280, 720, runs, base_a_ptr, base_b_ptr);
    benchmark(1920, 1080, runs, base_a_ptr, base_b_ptr);
    benchmark(3840, 2160, runs, base_a_ptr, base_b_ptr);
    return 0;
}
