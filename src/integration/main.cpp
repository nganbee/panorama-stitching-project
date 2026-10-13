// File: main.cpp
// Authors: Team 8
// Description: Command-line entry point for the sequential V0 pipeline.
//              Image I/O is outside the timed pipeline.
// External sources: stb_image and stb_image_write by Sean Barrett (public domain/MIT).
#include "../core_headers/versions.h"
#include "../libraries/stb_image.h"
#include "../libraries/stb_image_write.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>

namespace {

bool load_rgb(const char* path, Image& image) {
    int w = 0, h = 0, channels = 0;
    unsigned char* pixels = stbi_load(path, &w, &h, &channels, 3);
    if (pixels == nullptr) {
        std::fprintf(stderr, "Cannot load image: %s\n", path);
        return false;
    }
    image.w = w;
    image.h = h;
    image.c = 3;
    image.data.assign(pixels, pixels + static_cast<size_t>(w) * h * 3);
    stbi_image_free(pixels);
    return true;
}

bool save_rgb(const char* path, const Image& image) {
    if (image.w <= 0 || image.h <= 0 || image.data.empty()) return false;
    return stbi_write_png(path, image.w, image.h, 3, image.data.data(), image.w * 3) != 0;
}

void put_pixel(Image& image, int x, int y, uint8_t r, uint8_t g, uint8_t b) {
    if (x < 0 || x >= image.w || y < 0 || y >= image.h) return;
    const size_t index = (static_cast<size_t>(y) * image.w + x) * 3;
    image.data[index] = r;
    image.data[index + 1] = g;
    image.data[index + 2] = b;
}

void draw_line(Image& image, int x0, int y0, int x1, int y1,
               uint8_t r, uint8_t g, uint8_t b) {
    const int dx = std::abs(x1 - x0);
    const int sx = x0 < x1 ? 1 : -1;
    const int dy = -std::abs(y1 - y0);
    const int sy = y0 < y1 ? 1 : -1;
    int error = dx + dy;
    while (true) {
        put_pixel(image, x0, y0, r, g, b);
        if (x0 == x1 && y0 == y1) break;
        const int twice_error = 2 * error;
        if (twice_error >= dy) {
            error += dy;
            x0 += sx;
        }
        if (twice_error <= dx) {
            error += dx;
            y0 += sy;
        }
    }
}

void draw_cross(Image& image, int x, int y, uint8_t r, uint8_t g, uint8_t b) {
    for (int d = -3; d <= 3; ++d) {
        put_pixel(image, x + d, y, r, g, b);
        put_pixel(image, x, y + d, r, g, b);
    }
}

bool project_point(const Homography& H, double x, double y, double& px, double& py) {
    const double z = H.h[6] * x + H.h[7] * y + H.h[8];
    if (std::fabs(z) < 1e-12) return false;
    px = (H.h[0] * x + H.h[1] * y + H.h[2]) / z;
    py = (H.h[3] * x + H.h[4] * y + H.h[5]) / z;
    return std::isfinite(px) && std::isfinite(py);
}

bool is_inlier(const PipelineResult& result, const Match& match) {
    if (match.idxA < 0 || match.idxA >= static_cast<int>(result.keypointsA.size()) ||
        match.idxB < 0 || match.idxB >= static_cast<int>(result.keypointsB.size())) return false;
    double mapped_x = 0.0, mapped_y = 0.0;
    const Keypoint& source = result.keypointsB[match.idxB];
    const Keypoint& target = result.keypointsA[match.idxA];
    if (!project_point(result.H, source.x, source.y, mapped_x, mapped_y)) return false;
    const double dx = mapped_x - target.x;
    const double dy = mapped_y - target.y;
    return dx * dx + dy * dy <=
           static_cast<double>(params::kRansacThresh) * params::kRansacThresh;
}

bool save_keypoint_overlay(const std::string& path, const Image& source,
                           const std::vector<Keypoint>& keypoints) {
    Image overlay = source;
    for (const Keypoint& keypoint : keypoints)
        draw_cross(overlay, static_cast<int>(keypoint.x), static_cast<int>(keypoint.y), 255, 0, 0);
    return save_rgb(path.c_str(), overlay);
}

bool save_match_overlay(const std::string& path, const Image& A, const Image& B,
                        const PipelineResult& result) {
    Image overlay;
    overlay.w = A.w + B.w;
    overlay.h = std::max(A.h, B.h);
    overlay.c = 3;
    overlay.data.assign(static_cast<size_t>(overlay.w) * overlay.h * 3, 24);

    for (int y = 0; y < A.h; ++y) {
        const size_t src = static_cast<size_t>(y) * A.w * 3;
        const size_t dst = static_cast<size_t>(y) * overlay.w * 3;
        std::copy_n(A.data.begin() + static_cast<std::ptrdiff_t>(src),
                    static_cast<size_t>(A.w) * 3,
                    overlay.data.begin() + static_cast<std::ptrdiff_t>(dst));
    }
    for (int y = 0; y < B.h; ++y) {
        const size_t src = static_cast<size_t>(y) * B.w * 3;
        const size_t dst = (static_cast<size_t>(y) * overlay.w + A.w) * 3;
        std::copy_n(B.data.begin() + static_cast<std::ptrdiff_t>(src),
                    static_cast<size_t>(B.w) * 3,
                    overlay.data.begin() + static_cast<std::ptrdiff_t>(dst));
    }

    for (const Match& match : result.matches) {
        if (match.idxA < 0 || match.idxA >= static_cast<int>(result.keypointsA.size()) ||
            match.idxB < 0 || match.idxB >= static_cast<int>(result.keypointsB.size())) continue;
        const Keypoint& a = result.keypointsA[match.idxA];
        const Keypoint& b = result.keypointsB[match.idxB];
        const bool good = is_inlier(result, match);
        const uint8_t r = good ? 40 : 255;
        const uint8_t g = good ? 255 : 60;
        const uint8_t bl = 40;
        draw_line(overlay, static_cast<int>(a.x), static_cast<int>(a.y),
                  A.w + static_cast<int>(b.x), static_cast<int>(b.y), r, g, bl);
        draw_cross(overlay, static_cast<int>(a.x), static_cast<int>(a.y), r, g, bl);
        draw_cross(overlay, A.w + static_cast<int>(b.x), static_cast<int>(b.y), r, g, bl);
    }
    return save_rgb(path.c_str(), overlay);
}

bool save_debug_artifacts(const std::string& prefix, const Image& A, const Image& B,
                          const PipelineResult& result) {
    std::ofstream homography(prefix + "_homography.txt");
    if (!homography) return false;
    homography << "# H maps image B coordinates into image A coordinates\n";
    homography << "# rows: 3x3, normalized so H[2][2] = 1\n";
    homography << result.H.h[0] << ' ' << result.H.h[1] << ' ' << result.H.h[2] << '\n';
    homography << result.H.h[3] << ' ' << result.H.h[4] << ' ' << result.H.h[5] << '\n';
    homography << result.H.h[6] << ' ' << result.H.h[7] << ' ' << result.H.h[8] << '\n';
    homography << "keypoints_A=" << result.nKpA << "\n"
               << "keypoints_B=" << result.nKpB << "\n"
               << "matches=" << result.nMatches << "\n"
               << "inliers=" << result.nInliers << "\n"
               << "panorama=" << result.panorama.w << 'x' << result.panorama.h << "\n";

    return save_keypoint_overlay(prefix + "_keypoints_A.png", A, result.keypointsA) &&
           save_keypoint_overlay(prefix + "_keypoints_B.png", B, result.keypointsB) &&
           save_match_overlay(prefix + "_matches.png", A, B, result);
}

void print_usage(const char* program) {
    std::fprintf(stderr,
        "Usage: %s <left_image> <right_image> <output_image> "
        "--version 0 [--runs N] [--debug-prefix PREFIX]\n", program);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 6) {
        print_usage(argv[0]);
        return 2;
    }

    const char* left_path = argv[1];
    const char* right_path = argv[2];
    const char* output_path = argv[3];
    int version = -1;
    int runs = 5;
    std::string debug_prefix;

    for (int i = 4; i < argc; ++i) {
        if (std::strcmp(argv[i], "--version") == 0 && i + 1 < argc) {
            version = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--runs") == 0 && i + 1 < argc) {
            runs = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--debug-prefix") == 0 && i + 1 < argc) {
            debug_prefix = argv[++i];
        } else {
            print_usage(argv[0]);
            return 2;
        }
    }

    if (version < 0 || runs < 5) {
        std::fprintf(stderr, "--version is required and --runs must be at least 5.\n");
        return 2;
    }

    const PipelineFn pipeline = get_version(version);
    if (pipeline == nullptr) {
        std::fprintf(stderr, "Unsupported version: %d\n", version);
        return 2;
    }

    Image A, B;
    if (!load_rgb(left_path, A) || !load_rgb(right_path, B)) return 1;

    // Warm-up removes first-allocation effects from the reported averages.
    PipelineResult warmup;
    if (!pipeline(A, B, warmup)) {
        std::fprintf(stderr, "Warm-up pipeline failed: too few matches or invalid homography.\n");
        return 1;
    }

    StageTimes sum{};
    PipelineResult last;
    for (int run = 0; run < runs; ++run) {
        PipelineResult current;
        if (!pipeline(A, B, current)) {
            std::fprintf(stderr, "Pipeline failed on timed run %d.\n", run + 1);
            return 1;
        }
        sum.gray += current.times.gray;
        sum.blur += current.times.blur;
        sum.harris += current.times.harris;
        sum.nms_topk += current.times.nms_topk;
        sum.describe += current.times.describe;
        sum.match += current.times.match;
        sum.ransac += current.times.ransac;
        sum.warp_blend += current.times.warp_blend;
        sum.end_to_end += current.times.end_to_end;
        last = std::move(current);
    }

    const double inv_runs = 1.0 / static_cast<double>(runs);
    std::printf("version=%d runs=%d imageA=%dx%d imageB=%dx%d\n",
                version, runs, A.w, A.h, B.w, B.h);
    std::printf("keypoints_A=%d keypoints_B=%d matches=%d inliers=%d\n",
                last.nKpA, last.nKpB, last.nMatches, last.nInliers);
    std::printf("panorama=%dx%d inlier_ratio=%.3f\n",
                last.panorama.w, last.panorama.h,
                last.nMatches > 0 ? static_cast<double>(last.nInliers) / last.nMatches : 0.0);
    std::printf("avg_ms gray=%.3f blur=%.3f harris=%.3f nms_topk=%.3f "
                "describe=%.3f match=%.3f ransac=%.3f warp_blend=%.3f "
                "end_to_end=%.3f\n",
                sum.gray * inv_runs,
                sum.blur * inv_runs,
                sum.harris * inv_runs,
                sum.nms_topk * inv_runs,
                sum.describe * inv_runs,
                sum.match * inv_runs,
                sum.ransac * inv_runs,
                sum.warp_blend * inv_runs,
                sum.end_to_end * inv_runs);

    if (!save_rgb(output_path, last.panorama)) {
        std::fprintf(stderr, "Cannot save panorama: %s\n", output_path);
        return 1;
    }
    if (!debug_prefix.empty()) {
        // Collect diagnostics in a separate, untimed run so normal benchmark
        // timings do not include keypoint/match copies or artifact preparation.
        PipelineResult debug_result;
        debug_result.collect_debug = true;
        if (!pipeline(A, B, debug_result) ||
            !save_debug_artifacts(debug_prefix, A, B, debug_result)) {
            std::fprintf(stderr, "Cannot save debug artifacts with prefix: %s\n", debug_prefix.c_str());
            return 1;
        }
    }
    return 0;
}
