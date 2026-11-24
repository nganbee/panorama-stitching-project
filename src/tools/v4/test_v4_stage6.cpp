// Authors: Team 8; implementation assisted by Antigravity.
// Isolated S6 warp/blend/crop parity: identity, translation, perspective, partial blocks, and invalid inputs.
#include "../../core_headers/v0_stages.h"
#include "../../core_headers/v1_stages.h"
#include "../../core_headers/v3_stages.h"
#include "../../core_headers/v4_stages.h"
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

Image make_pattern_image(int w, int h, int pattern, std::mt19937& rng) {
    Image img;
    img.w = w;
    img.h = h;
    img.c = 3;
    img.data.resize(static_cast<size_t>(w) * h * 3);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const size_t idx = (static_cast<size_t>(y) * w + x) * 3;
            if (pattern == 0) {
                img.data[idx] = 128; img.data[idx + 1] = 64; img.data[idx + 2] = 200;
            } else if (pattern == 1) {
                const uint8_t v = ((x + y) % 2) ? 255 : 0;
                img.data[idx] = v; img.data[idx + 1] = 255 - v; img.data[idx + 2] = v;
            } else if (pattern == 2) {
                img.data[idx]     = static_cast<uint8_t>((x * 255) / (w > 1 ? w - 1 : 1));
                img.data[idx + 1] = static_cast<uint8_t>((y * 255) / (h > 1 ? h - 1 : 1));
                img.data[idx + 2] = static_cast<uint8_t>(((x + y) * 127) / (w + h > 2 ? w + h - 2 : 1));
            } else {
                img.data[idx]     = static_cast<uint8_t>(rng() & 255u);
                img.data[idx + 1] = static_cast<uint8_t>(rng() & 255u);
                img.data[idx + 2] = static_cast<uint8_t>(rng() & 255u);
            }
        }
    }
    return img;
}

void compare_pano(const Image& ref, const Image& actual, const char* label) {
    require(ref.w == actual.w && ref.h == actual.h && ref.data.size() == actual.data.size(), label);
    for (size_t i = 0; i < ref.data.size(); ++i) {
        const int err = std::abs(int(ref.data[i]) - int(actual.data[i]));
        require(err <= 1, label);
    }
}

void check_warp(const Image& A, const Image& B, const Homography& H, const char* label) {
    Image cpu, v3, v4;
    v0_warp_blend(A, B, H, cpu);
    v3_warp_blend(A, B, H, v3);
    v4_warp_blend(A, B, H, v4);
    require(valid_rgb_image(cpu) && valid_rgb_image(v3) && valid_rgb_image(v4),
            "Positive S6 case returned an empty image; parity is NOT verified");
    compare_pano(cpu, v4, label);
    compare_pano(v3, v4, label);
}
} // namespace

int main() {
    try {
        std::mt19937 rng(741);
        const int shapes[][2] = {
            {1, 1}, {7, 5}, {16, 16}, {17, 19}, {31, 33}, {32, 8}, {33, 9}, {193, 157}
        };
        const Homography identity{{1, 0, 0, 0, 1, 0, 0, 0, 1}};
        const Homography shift{{1, 0, 12.25, 0, 1, -5.5, 0, 0, 1}};
        const Homography perspective{{0.98, -0.04, 15.3, 0.03, 0.99, -7.2, -0.0002, 0.0001, 1.0}};

        for (const auto& s : shapes) {
            for (int pattern = 0; pattern < 4; ++pattern) {
                const Image A = make_pattern_image(s[0], s[1], pattern, rng);
                const Image B = make_pattern_image(s[0], s[1], (pattern + 1) % 4, rng);
                check_warp(A, B, identity, "V4 S6 identity parity failed");
                check_warp(A, B, shift, "V4 S6 shift parity failed");
                if (s[0] >= 16 && s[1] >= 16) {
                    check_warp(A, B, perspective, "V4 S6 perspective parity failed");
                }
            }
        }

        // Invalid inputs and degenerate homography must return empty output cleanly.
        Image out;
        const Image valid_img = make_pattern_image(32, 32, 2, rng);
        v4_warp_blend(Image{}, valid_img, identity, out);
        require(out.data.empty(), "Empty image A accepted");
        v4_warp_blend(valid_img, Image{}, identity, out);
        require(out.data.empty(), "Empty image B accepted");
        const Homography singular{{0, 0, 0, 0, 0, 0, 0, 0, 0}};
        v4_warp_blend(valid_img, valid_img, singular, out);
        require(out.data.empty(), "Singular homography accepted");

        std::puts("PASS: V4 S6 warp vs V0/V3 (identity, shift, perspective, partial blocks, invalid inputs)");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "V4 S6 warp FAIL: %s\n", e.what());
        return 1;
    }
}
