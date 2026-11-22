// Authors: Team 8; V2 verification assisted by OpenAI Codex.
// External sources: none. Includes original V0 once to observe its private math.
// Link this TU INSTEAD OF v0_stage6.cpp in the verification binary only.
#include "s6_diagnostics.h"
#include "../../src/stage_implementations/v0_stage6.cpp"

V2S6Diagnostics v2_cpu_s6_diagnostics(const Image& A, const Image& B, const Homography& H) {
    V2S6Diagnostics out;
    double h[9], inverse[9];
    if (!valid_rgb_image(A) || !valid_rgb_image(B) || !prepare_transform(H, h, inverse)) return out;
    double min_x = 0, min_y = 0, max_x = A.w - 1.0, max_y = A.h - 1.0;
    int previous_sign = 0;
    for (int i = 0; i < 4; ++i) {
        double x, y;
        const double bx = (i & 1) ? B.w - 1.0 : 0, by = (i & 2) ? B.h - 1.0 : 0;
        if (!project(h, bx, by, x, y)) return out;
        const int sign = h[6] * bx + h[7] * by + h[8] > 0 ? 1 : -1;
        if (previous_sign && previous_sign != sign) return out;
        previous_sign = sign;
        min_x = std::min(min_x, snap_integer(x)); min_y = std::min(min_y, snap_integer(y));
        max_x = std::max(max_x, snap_integer(x)); max_y = std::max(max_y, snap_integer(y));
    }
    min_x = std::floor(min_x); min_y = std::floor(min_y);
    max_x = std::ceil(max_x); max_y = std::ceil(max_y);
    const double w = max_x - min_x + 1, ht = max_y - min_y + 1;
    if (w <= 0 || ht <= 0 || w > kMaxCanvasDimension || ht > kMaxCanvasDimension ||
        min_x < std::numeric_limits<int>::min() || min_y < std::numeric_limits<int>::min() ||
        max_x > std::numeric_limits<int>::max() || max_y > std::numeric_limits<int>::max()) return out;
    out.width = int(w); out.height = int(ht); out.origin_x = int(min_x); out.origin_y = int(min_y);
    const size_t pixels = size_t(out.width) * out.height;
    if (pixels > kMaxCanvasPixels) return out;
    out.mask.resize(pixels);
    const double tolerance = 1e-7 + kRoundoffFactor * std::max(B.w, B.h);
    for (int y = 0; y < out.height; ++y) for (int x = 0; x < out.width; ++x) {
        const int ax = x + out.origin_x, ay = y + out.origin_y;
        const bool in_a = ax >= 0 && ax < A.w && ay >= 0 && ay < A.h;
        double bx, by;
        const bool in_b = project(inverse, ax, ay, bx, by) && bx >= -tolerance && by >= -tolerance &&
                          bx <= B.w - 1.0 + tolerance && by <= B.h - 1.0 + tolerance;
        out.mask[size_t(y) * out.width + x] = in_a || in_b;
    }
    const Rect crop = largest_valid_rectangle(out.mask, out.width, out.height);
    out.crop_x = crop.x; out.crop_y = crop.y; out.crop_w = crop.w; out.crop_h = crop.h;
    out.ok = crop.w > 0 && crop.h > 0;
    return out;
}
