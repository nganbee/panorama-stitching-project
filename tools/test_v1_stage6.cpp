// File: test_v1_stage6.cpp
// Authors: Team 8
// Description: Verification test comparing V0 (CPU) and V1 (CUDA) Stage 6 results.
// External sources: none
#include "../src/core_headers/common.h"
#include "../src/core_headers/v0_stages.h"
#include "../src/core_headers/v1_stages.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const std::string& name) {
    std::cout << (condition ? "[PASS] " : "[FAIL] ") << name << "\n";
    if (!condition) ++failures;
}

Image solid(int w, int h, uint8_t value) {
    Image image;
    image.w = w; image.h = h; image.c = 3;
    image.data.assign(static_cast<size_t>(w) * h * 3, value);
    return image;
}

Homography translation(double x = 0.0, double y = 0.0) {
    return Homography{{1.0, 0.0, x, 0.0, 1.0, y, 0.0, 0.0, 1.0}};
}

void compare_v0_v1(const Image& A, const Image& B, const Homography& H, const std::string& test_name) {
    Image pano_v0, pano_v1;
    v0_warp_blend(A, B, H, pano_v0);
    v1_warp_blend(A, B, H, pano_v1);

    bool match = (pano_v0.w == pano_v1.w) &&
                 (pano_v0.h == pano_v1.h) &&
                 (pano_v0.c == pano_v1.c) &&
                 (pano_v0.data == pano_v1.data);

    if (!match) {
        std::cerr << "Mismatch details for " << test_name << ":\n";
        std::cerr << "  V0 size: " << pano_v0.w << "x" << pano_v0.h << ", bytes: " << pano_v0.data.size() << "\n";
        std::cerr << "  V1 size: " << pano_v1.w << "x" << pano_v1.h << ", bytes: " << pano_v1.data.size() << "\n";
        if (pano_v0.data.size() == pano_v1.data.size()) {
            size_t diff_count = 0;
            int max_diff = 0;
            for (size_t i = 0; i < pano_v0.data.size(); ++i) {
                int d = std::abs(int(pano_v0.data[i]) - int(pano_v1.data[i]));
                if (d > 0) {
                    diff_count++;
                    max_diff = std::max(max_diff, d);
                }
            }
            std::cerr << "  Diff count: " << diff_count << "/" << pano_v0.data.size()
                      << ", Max diff: " << max_diff << "\n";
        }
    }
    check(match, test_name);
}

}  // namespace

int main() {
    std::cout << "=== Running V0 vs V1 Stage 6 Verification ===\n";

    // 1. Identity transform
    const Image same = solid(10, 10, 237);
    compare_v0_v1(same, same, translation(), "1. Identity transform with solid color");

    // 2. Pattern image
    Image varied = solid(20, 15, 0);
    for (size_t i = 0; i < varied.data.size(); ++i) {
        varied.data[i] = static_cast<uint8_t>((i * 73) % 256);
    }
    compare_v0_v1(varied, varied, translation(), "2. Identity transform with varying RGB pattern");

    // 3. Translation right & down
    const Image A = solid(20, 20, 80), B = solid(20, 20, 200);
    compare_v0_v1(A, B, translation(6, 0), "3. Horizontal translation (right)");
    compare_v0_v1(A, B, translation(0, 6), "4. Vertical translation (down)");
    compare_v0_v1(A, B, translation(-5, -5), "5. Diagonal translation (left & up)");

    // 4. Subpixel fractional translation (bilinear interpolation test)
    Image gradient = solid(16, 16, 0);
    for (int y = 0; y < gradient.h; ++y) {
        for (int x = 0; x < gradient.w; ++x) {
            for (int c = 0; c < 3; ++c) {
                gradient.data[(static_cast<size_t>(y) * gradient.w + x) * 3 + c] =
                    static_cast<uint8_t>(10 * x + 5 * y + c * 20);
            }
        }
    }
    compare_v0_v1(A, gradient, translation(2.5, 3.25), "6. Fractional subpixel translation (Bilinear)");

    // 5. Perspective transformation
    Homography perspective{{1.0, 0.05, 4.0, 0.02, 1.0, -2.0, 0.0005, 0.0002, 1.0}};
    compare_v0_v1(varied, gradient, perspective, "7. Perspective projective warp with feather blending");

    // 6. Degenerate / invalid inputs
    Homography invalid = translation();
    invalid.h[0] = 0.0; invalid.h[4] = 0.0; // Singular
    compare_v0_v1(A, B, invalid, "8. Rejection of singular homography");

    // Cover multiple histogram/row blocks and >256 reduction candidates.
    Image large = solid(257, 263, 0);
    for (size_t i = 0; i < large.data.size(); ++i) {
        large.data[i] = static_cast<uint8_t>((i * 73 + i / 17) % 256);
    }
    compare_v0_v1(large, large, translation(), "9. Multiple blocks and reduction groups");
    compare_v0_v1(large, large, translation(63.5, -7.25), "10. Large fractional translation");
    compare_v0_v1(large, large, perspective, "11. Large perspective crop");
    compare_v0_v1(solid(1, 513, 80), solid(1, 513, 200), translation(), "12. Single column");
    compare_v0_v1(solid(513, 1, 80), solid(513, 1, 200), translation(), "13. Single row");
    compare_v0_v1(A, B, translation(50, 0), "14. Equal-area crops prefer leftmost");
    compare_v0_v1(A, B, translation(0, 50), "15. Equal-area crops prefer earliest row");

    std::cout << "\nResult: " << (failures == 0 ? "ALL PASS" : "FAILURES DETECTED")
              << " (" << failures << " failures)\n";
    return failures == 0 ? 0 : 1;
}

