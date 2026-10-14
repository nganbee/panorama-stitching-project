// File: v0_stage6.cpp
// Authors: Team 8
// Description: Sequential inverse bilinear warp, feather blend and valid crop.
//              Bounds and numerical guards reviewed with Codex assistance.
// External sources: none
#include "../core_headers/common.h"
#include "../core_headers/v0_stages.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

// Prevent a corrupt/near-horizon transform from requesting an unbounded canvas.
// This permits 8K panoramas and caps RGB + validity storage at 256 MiB.
constexpr int kMaxCanvasDimension = 32768;
constexpr size_t kMaxCanvasPixels = size_t{64} * 1024 * 1024;
constexpr double kRoundoffFactor = 64.0 * std::numeric_limits<double>::epsilon();

struct Rect {
    int x = 0;
    int y = 0;
    int w = 0;
    int h = 0;
};

bool project(const double h[9], double x, double y, double& out_x, double& out_y) {
    const double z = h[6] * x + h[7] * y + h[8];
    const double scale = std::abs(h[6] * x) + std::abs(h[7] * y) + std::abs(h[8]);
    if (!std::isfinite(z) || z == 0.0 || std::abs(z) <= kRoundoffFactor * scale) return false;
    out_x = (h[0] * x + h[1] * y + h[2]) / z;
    out_y = (h[3] * x + h[4] * y + h[5]) / z;
    return std::isfinite(out_x) && std::isfinite(out_y);
}

double snap_integer(double value) {
    const double nearest = std::round(value);
    return std::abs(value - nearest) <= 1e-8 + kRoundoffFactor * std::abs(value)
        ? nearest : value;
}

bool prepare_transform(const Homography& H, double h[9], double inverse[9]) {
    double scale = 0.0;
    for (double value : H.h) {
        if (!std::isfinite(value)) return false;
        scale = std::max(scale, std::abs(value));
    }
    if (scale == 0.0) return false;
    for (int i = 0; i < 9; ++i) h[i] = H.h[i] / scale;

    // Adjugate is a projectively equivalent inverse; division by det is unnecessary.
    inverse[0] = h[4] * h[8] - h[5] * h[7];
    inverse[1] = h[2] * h[7] - h[1] * h[8];
    inverse[2] = h[1] * h[5] - h[2] * h[4];
    inverse[3] = h[5] * h[6] - h[3] * h[8];
    inverse[4] = h[0] * h[8] - h[2] * h[6];
    inverse[5] = h[2] * h[3] - h[0] * h[5];
    inverse[6] = h[3] * h[7] - h[4] * h[6];
    inverse[7] = h[1] * h[6] - h[0] * h[7];
    inverse[8] = h[0] * h[4] - h[1] * h[3];

    const double t0 = h[0] * inverse[0];
    const double t1 = h[1] * inverse[3];
    const double t2 = h[2] * inverse[6];
    const double determinant = t0 + t1 + t2;
    const double determinant_scale = std::abs(t0) + std::abs(t1) + std::abs(t2);
    if (!std::isfinite(determinant) || determinant == 0.0 ||
        std::abs(determinant) <= kRoundoffFactor * determinant_scale) return false;
    return true;
}

// Largest all-valid axis-aligned rectangle, using a histogram/monotonic stack.
// Ties prefer a wider rectangle, then the first rectangle encountered.
Rect largest_valid_rectangle(const std::vector<uint8_t>& valid, int width, int height) {
    Rect best;
    std::vector<int> heights(static_cast<size_t>(width), 0);
    std::vector<int> stack;
    stack.reserve(static_cast<size_t>(width) + 1);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            heights[x] = valid[static_cast<size_t>(y) * width + x] ? heights[x] + 1 : 0;
        }
        stack.clear();
        for (int x = 0; x <= width; ++x) {
            const int current = x < width ? heights[x] : 0;
            while (!stack.empty() && current < heights[stack.back()]) {
                const int bar = stack.back();
                stack.pop_back();
                const int left = stack.empty() ? 0 : stack.back() + 1;
                const int rectangle_width = x - left;
                const int rectangle_height = heights[bar];
                const long long area = static_cast<long long>(rectangle_width) * rectangle_height;
                const long long best_area = static_cast<long long>(best.w) * best.h;
                if (area > best_area || (area == best_area && rectangle_width > best.w)) {
                    best = {left, y - rectangle_height + 1, rectangle_width, rectangle_height};
                }
            }
            if (x < width) stack.push_back(x);
        }
    }
    return best;
}

void crop_image(Image& image, const Rect& rect) {
    if (rect.x == 0 && rect.y == 0 && rect.w == image.w && rect.h == image.h) return;
    std::vector<uint8_t> cropped(static_cast<size_t>(rect.w) * rect.h * 3);
    for (int y = 0; y < rect.h; ++y) {
        const size_t src = (static_cast<size_t>(rect.y + y) * image.w + rect.x) * 3;
        const size_t dst = static_cast<size_t>(y) * rect.w * 3;
        std::copy_n(image.data.begin() + static_cast<std::ptrdiff_t>(src),
                    static_cast<size_t>(rect.w) * 3,
                    cropped.begin() + static_cast<std::ptrdiff_t>(dst));
    }
    image.w = rect.w;
    image.h = rect.h;
    image.data = std::move(cropped);
}

uint8_t rounded_channel(double value) {
    return static_cast<uint8_t>(std::lround(std::max(0.0, std::min(255.0, value))));
}

}  // namespace

void v0_warp_blend(const Image& A, const Image& B, const Homography& H, Image& pano) {
    // Build locally so output may safely alias either source image.
    Image result;
    auto fail = [&]() { pano = Image{}; };
    if (!valid_rgb_image(A) || !valid_rgb_image(B)) { fail(); return; }
    double h[9], inverse[9];
    if (!prepare_transform(H, h, inverse)) { fail(); return; }

    double min_x = 0.0, min_y = 0.0;
    double max_x = A.w - 1.0, max_y = A.h - 1.0;
    int denominator_sign = 0;
    for (int corner = 0; corner < 4; ++corner) {
        const double bx = (corner & 1) ? B.w - 1.0 : 0.0;
        const double by = (corner & 2) ? B.h - 1.0 : 0.0;
        double ax, ay;
        if (!project(h, bx, by, ax, ay)) { fail(); return; }
        const double z = h[6] * bx + h[7] * by + h[8];
        const int sign = z > 0.0 ? 1 : -1;
        // A linear denominator with different corner signs crosses a horizon.
        if (denominator_sign != 0 && denominator_sign != sign) { fail(); return; }
        denominator_sign = sign;
        ax = snap_integer(ax);
        ay = snap_integer(ay);
        min_x = std::min(min_x, ax);
        min_y = std::min(min_y, ay);
        max_x = std::max(max_x, ax);
        max_y = std::max(max_y, ay);
    }
    min_x = std::floor(min_x);
    min_y = std::floor(min_y);
    max_x = std::ceil(max_x);
    max_y = std::ceil(max_y);
    // Bounds refer to inclusive pixel centers, hence the final +1.
    const double width = max_x - min_x + 1.0;
    const double height = max_y - min_y + 1.0;
    if (!std::isfinite(width) || !std::isfinite(height) ||
        width <= 0.0 || height <= 0.0 ||
        width > kMaxCanvasDimension || height > kMaxCanvasDimension ||
        min_x < std::numeric_limits<int>::min() || min_y < std::numeric_limits<int>::min() ||
        max_x > std::numeric_limits<int>::max() || max_y > std::numeric_limits<int>::max()) {
        fail(); return;
    }
    const int origin_x = static_cast<int>(min_x), origin_y = static_cast<int>(min_y);
    const int pano_w = static_cast<int>(width), pano_h = static_cast<int>(height);
    const size_t pixels = static_cast<size_t>(pano_w) * pano_h;
    if (pixels > kMaxCanvasPixels) { fail(); return; }

    try {
        result.w = pano_w;
        result.h = pano_h;
        result.data.assign(pixels * 3, 0);
        std::vector<uint8_t> valid(pixels, 0);
        const double sample_tolerance = 1e-7 + kRoundoffFactor * std::max(B.w, B.h);
        for (int py = 0; py < pano_h; ++py) {
            for (int px = 0; px < pano_w; ++px) {
                const int ax = px + origin_x, ay = py + origin_y;
                const bool in_A = ax >= 0 && ax < A.w && ay >= 0 && ay < A.h;
                double bx = 0.0, by = 0.0;
                const bool in_B = project(inverse, ax, ay, bx, by) &&
                    bx >= -sample_tolerance && bx <= B.w - 1.0 + sample_tolerance &&
                    by >= -sample_tolerance && by <= B.h - 1.0 + sample_tolerance;
                double color_B[3] = {};
                if (in_B) {
                    // Clamp after the tolerance check: roundoff must not discard
                    // a valid last row/column or cause a negative source index.
                    bx = std::max(0.0, std::min(B.w - 1.0, bx));
                    by = std::max(0.0, std::min(B.h - 1.0, by));
                    const int x0 = static_cast<int>(std::floor(bx));
                    const int y0 = static_cast<int>(std::floor(by));
                    const int x1 = std::min(x0 + 1, B.w - 1);
                    const int y1 = std::min(y0 + 1, B.h - 1);
                    const double dx = bx - x0, dy = by - y0;
                    for (int c = 0; c < 3; ++c) {
                        const double v00 = B.data[(static_cast<size_t>(y0) * B.w + x0) * 3 + c];
                        const double v10 = B.data[(static_cast<size_t>(y0) * B.w + x1) * 3 + c];
                        const double v01 = B.data[(static_cast<size_t>(y1) * B.w + x0) * 3 + c];
                        const double v11 = B.data[(static_cast<size_t>(y1) * B.w + x1) * 3 + c];
                        color_B[c] = (v00 * (1.0 - dx) + v10 * dx) * (1.0 - dy) +
                                     (v01 * (1.0 - dx) + v11 * dx) * dy;
                    }
                }

                const size_t pixel = static_cast<size_t>(py) * pano_w + px;
                const size_t source_A = in_A ? (static_cast<size_t>(ay) * A.w + ax) * 3 : 0;
                double weight_A = in_A ? 1.0 : 0.0;
                double weight_B = in_B ? 1.0 : 0.0;
                if (in_A && in_B) {
                    const double distance_A = std::min({double(ax), A.w - 1.0 - ax,
                                                        double(ay), A.h - 1.0 - ay});
                    const double distance_B = std::min({bx, B.w - 1.0 - bx, by, B.h - 1.0 - by});
                    const double total = distance_A + distance_B;
                    if (total > 0.0) {
                        weight_A = distance_A / total;
                        weight_B = distance_B / total;
                    } else {
                        // Both contributors remain valid when both border distances
                        // are zero. Equal weights preserve a colored/black border.
                        weight_A = weight_B = 0.5;
                    }
                }
                for (int c = 0; c < 3; ++c) {
                    const double color_A = in_A ? A.data[source_A + c] : 0.0;
                    result.data[pixel * 3 + c] = rounded_channel(color_A * weight_A + color_B[c] * weight_B);
                }
                valid[pixel] = in_A || in_B ? 1 : 0;
            }
        }
        // Validity is geometric: real black input pixels must survive cropping.
        const Rect crop = largest_valid_rectangle(valid, pano_w, pano_h);
        if (crop.w <= 0 || crop.h <= 0) { fail(); return; }
        crop_image(result, crop);
        pano = std::move(result);
    } catch (const std::bad_alloc&) {
        fail();
    } catch (const std::length_error&) {
        fail();
    }
}
