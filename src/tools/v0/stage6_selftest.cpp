// File: stage6_selftest.cpp
// Authors: Team 8
// Description: Regression checks for warp bounds, bilinear sampling, valid crop,
//              feather border normalization and fail-safe malformed inputs.
//              Test cases reviewed with Codex assistance.
// External sources: none
#include "../src/core_headers/v0_stages.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>

namespace {
int failures = 0;
void check(bool condition, const std::string& name) {
    std::cout << (condition ? "[PASS] " : "[FAIL] ") << name << "\n";
    if (!condition) ++failures;
}
Image solid(int w, int h, uint8_t value) {
    Image image;
    image.w = w; image.h = h;
    image.data.assign(static_cast<size_t>(w) * h * 3, value);
    return image;
}
Homography translation(double x = 0.0, double y = 0.0) {
    return Homography{{1.0, 0.0, x, 0.0, 1.0, y, 0.0, 0.0, 1.0}};
}
uint8_t pixel(const Image& image, int x, int y, int c = 0) {
    return image.data[(static_cast<size_t>(y) * image.w + x) * 3 + c];
}
bool empty(const Image& image) {
    return image.w == 0 && image.h == 0 && image.data.empty();
}
void expect_empty(const Image& A, const Image& B, const Homography& H,
                  const std::string& name) {
    Image output = solid(2, 2, 123); // Must clear stale output on failure.
    v0_warp_blend(A, B, H, output);
    check(empty(output), name);
}
}  // namespace

int main() {
    const Image same = solid(10, 10, 237);
    Image output;
    v0_warp_blend(same, same, translation(), output);
    check(output.w == 10 && output.h == 10 && output.data == same.data,
          "identity preserves every pixel, including all four borders");

    Image varied = solid(7, 5, 0);
    for (size_t i = 0; i < varied.data.size(); ++i) {
        varied.data[i] = static_cast<uint8_t>((i * 73) % 256);
    }
    v0_warp_blend(varied, varied, translation(), output);
    check(output.w == varied.w && output.h == varied.h && output.data == varied.data,
          "identity preserves arbitrary RGB values without quantization drift");

    for (const auto dims : {std::pair<int, int>{1, 1}, {1, 6}, {6, 1}}) {
        const Image thin = solid(dims.first, dims.second, 254);
        v0_warp_blend(thin, thin, translation(), output);
        check(output.w == thin.w && output.h == thin.h && output.data == thin.data,
              "degenerate image dimension " + std::to_string(thin.w) + "x" + std::to_string(thin.h));
    }

    const Image A = solid(10, 10, 80), B = solid(10, 10, 200);
    v0_warp_blend(A, B, translation(4, 0), output);
    check(output.w == 14 && output.h == 10 && pixel(output, 13, 0) == 200 &&
          pixel(output, 13, 9) == 200 && pixel(output, 0, 0) == 80,
          "right translation includes final B column and source borders");
    v0_warp_blend(A, B, translation(0, 4), output);
    check(output.w == 10 && output.h == 14 && pixel(output, 0, 13) == 200 &&
          pixel(output, 9, 13) == 200 && pixel(output, 0, 0) == 80,
          "down translation includes final B row and source borders");
    v0_warp_blend(A, B, translation(-4, 0), output);
    check(output.w == 14 && output.h == 10 && pixel(output, 0, 0) == 200 &&
          pixel(output, 13, 9) == 80, "left translation preserves both outer columns");
    v0_warp_blend(A, B, translation(0, -4), output);
    check(output.w == 10 && output.h == 14 && pixel(output, 0, 0) == 200 &&
          pixel(output, 9, 13) == 80, "up translation preserves both outer rows");
    v0_warp_blend(A, B, translation(), output);
    check(output.w == 10 && output.h == 10 && pixel(output, 0, 0) == 140 &&
          pixel(output, 9, 9) == 140, "zero-distance feather weights average two valid borders");

    const Image black = solid(10, 10, 0);
    v0_warp_blend(black, black, translation(4, 0), output);
    check(output.w == 14 && output.h == 10 &&
          std::all_of(output.data.begin(), output.data.end(), [](uint8_t v) { return v == 0; }),
          "real black source pixels remain geometrically valid during crop");

    const Image small_A = solid(4, 4, 0);
    Image gradient = solid(4, 4, 0);
    for (int y = 0; y < gradient.h; ++y) {
        for (int x = 0; x < gradient.w; ++x) {
            for (int c = 0; c < 3; ++c) {
                gradient.data[(static_cast<size_t>(y) * gradient.w + x) * 3 + c] =
                    static_cast<uint8_t>(40 * x + 10 * y);
            }
        }
    }
    v0_warp_blend(small_A, gradient, translation(2.5, 0), output);
    check(output.w == 6 && output.h == 4 && pixel(output, 4, 1) == 70 &&
          pixel(output, 5, 1) == 110,
          "fractional translation uses bilinear values on B-only pixels");
    v0_warp_blend(small_A, gradient, translation(0.5, 0.5), output);
    check(output.w == 4 && output.h == 4 && pixel(output, 2, 2) == 45,
          "subpixel warp and normalized interior feather weights agree");

    Homography shear = translation(2, 0);
    shear.h[1] = 0.5;
    const Image shear_source = solid(4, 4, 155);
    v0_warp_blend(shear_source, shear_source, shear, output);
    check(output.w == 6 && output.h == 4 &&
          std::all_of(output.data.begin(), output.data.end(), [](uint8_t v) { return v == 155; }),
          "largest valid rectangle removes only geometrically empty warped corners");

    for (double error : {-1e-10, 1e-10}) {
        v0_warp_blend(A, B, translation(4 + error, 0), output);
        check(output.w == 14 && output.h == 10 && pixel(output, 13, 9) == 200,
              "integer boundary is stable under floating-point roundoff");
    }
    for (double scale : {1e200, 1e-200, -3.0}) {
        Homography H = translation();
        for (double& value : H.h) value *= scale;
        v0_warp_blend(varied, varied, H, output);
        check(output.w == varied.w && output.h == varied.h && output.data == varied.data,
              "homography is invariant to finite nonzero scalar normalization");
    }
    Image alias = varied;
    v0_warp_blend(alias, alias, translation(), alias);
    check(alias.data == varied.data && alias.w == varied.w && alias.h == varied.h,
          "output may alias a source without destroying input");

    Homography invalid = translation();
    invalid.h[0] = std::numeric_limits<double>::quiet_NaN();
    expect_empty(A, B, invalid, "reject nonfinite homography");
    invalid = translation();
    invalid.h[8] = std::numeric_limits<double>::infinity();
    expect_empty(A, B, invalid, "reject infinite homography");
    invalid = Homography{};
    expect_empty(A, B, invalid, "reject zero homography");
    invalid = translation();
    invalid.h[4] = 0.0;
    expect_empty(A, B, invalid, "reject singular homography");
    invalid = translation();
    invalid.h[6] = 1.0; invalid.h[8] = -4.0;
    expect_empty(A, B, invalid, "reject a projective horizon crossing B");
    invalid = translation();
    invalid.h[6] = 1.0; invalid.h[8] = 0.0;
    expect_empty(A, B, invalid, "reject a projective horizon touching a B corner");
    expect_empty(A, B, translation(1e100, 0), "reject unsafe huge canvas before allocation");

    Image bad = A;
    bad.c = 1;
    expect_empty(bad, B, translation(), "reject non-RGB input");
    bad = A; bad.data.pop_back();
    expect_empty(A, bad, translation(), "reject truncated source vector");
    bad = A; bad.data.push_back(0);
    expect_empty(bad, B, translation(), "reject oversized inconsistent source vector");
    bad = A; bad.w = -1;
    expect_empty(A, bad, translation(), "reject negative image dimension");
    expect_empty(Image{}, B, translation(), "reject empty input image");

    std::cout << (failures == 0 ? "PASS" : "FAIL") << ": Stage 6 regression suite ("
              << failures << " failures)\n";
    return failures == 0 ? 0 : 1;
}

