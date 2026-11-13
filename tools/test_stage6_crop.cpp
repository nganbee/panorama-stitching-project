// Portable correctness test for the same row scan and comparator used on GPU.
// Build: g++ -O2 -std=c++17 tools/test_stage6_crop.cpp -o test_stage6_crop
#define PANORAMA_STAGE6_HOST_TEST
#include "../src/stage_implementations/v1_stage6.cu"
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using stage6_crop::Rect;

// Independent brute-force oracle. Enumerate bottom edge then right edge in
// V0's encounter order, and only replace on larger area or larger width.
Rect brute_force(const std::vector<uint8_t>& valid, int w, int h) {
    const int stride = w + 1;
    std::vector<int> sums(static_cast<size_t>(stride) * (h + 1), 0);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            sums[(y + 1) * stride + x + 1] = valid[y * w + x] +
                sums[y * stride + x + 1] + sums[(y + 1) * stride + x] - sums[y * stride + x];
        }
    }
    Rect best{};
    for (int bottom = 1; bottom <= h; ++bottom) {
        for (int right = 1; right <= w; ++right) {
            for (int top = 0; top < bottom; ++top) {
                for (int left = 0; left < right; ++left) {
                    const int area = (right - left) * (bottom - top);
                    const int sum = sums[bottom * stride + right] - sums[top * stride + right] -
                        sums[bottom * stride + left] + sums[top * stride + left];
                    const int best_area = best.w * best.h;
                    if (sum == area && (area > best_area ||
                        (area == best_area && right - left > best.w))) {
                        best = {left, top, right - left, bottom - top};
                    }
                }
            }
        }
    }
    return best;
}

Rect reduce_rows(const std::vector<Rect>& rows) {
    Rect candidates[256]{};
    for (size_t lane = 0; lane < 256; ++lane) {
        for (size_t y = lane; y < rows.size(); y += 256) {
            if (stage6_crop::better(rows[y], candidates[lane])) candidates[lane] = rows[y];
        }
    }
    for (int stride = 128; stride > 0; stride /= 2) {
        for (int lane = 0; lane < stride; ++lane) {
            if (stage6_crop::better(candidates[lane + stride], candidates[lane])) {
                candidates[lane] = candidates[lane + stride];
            }
        }
    }
    return candidates[0];
}

void check(const std::vector<uint8_t>& mask, int w, int h) {
    std::vector<int> heights(mask.size()), stacks(mask.size());
    std::vector<Rect> rows(h);
    for (int x = 0; x < w; ++x) {
        int count = 0;
        for (int y = 0; y < h; ++y) {
            const size_t pixel = static_cast<size_t>(y) * w + x;
            heights[pixel] = count = mask[pixel] ? count + 1 : 0;
        }
    }
    for (int y = 0; y < h; ++y) {
        const size_t offset = static_cast<size_t>(y) * w;
        rows[y] = stage6_crop::scan_row(heights.data() + offset, w, y, stacks.data() + offset);
    }
    const Rect expected = brute_force(mask, w, h);
    // Reversing completion order must not change the winner, including ties.
    for (int order = 0; order < 2; ++order) {
        const Rect actual = reduce_rows(rows);
        if (actual.x != expected.x || actual.y != expected.y ||
            actual.w != expected.w || actual.h != expected.h) {
            throw std::runtime_error("Crop mismatch at " + std::to_string(w) + "x" + std::to_string(h));
        }
        std::reverse(rows.begin(), rows.end());
    }
}
} // namespace

int main() {
    try {
        // The same transform/canvas helper runs inside prepare_canvas_kernel.
        const Homography identity{{1,0,0,0,1,0,0,0,1}};
        CanvasConfig canvas = make_canvas(17, 19, 17, 19, identity);
        if (!canvas.ok || canvas.width != 17 || canvas.height != 19 || canvas.origin_x || canvas.origin_y)
            throw std::runtime_error("Identity canvas mismatch");
        const Homography translated{{1,0,-3.5,0,1,2.25,0,0,1}};
        canvas = make_canvas(17, 19, 17, 19, translated);
        if (!canvas.ok || canvas.width != 21 || canvas.height != 22 || canvas.origin_x != -4 || canvas.origin_y != 0)
            throw std::runtime_error("Fractional canvas mismatch");
        Homography singular{};
        if (make_canvas(17,19,17,19,singular).ok) throw std::runtime_error("Singular H accepted");
        Homography horizon{{1,0,0,0,1,0,-0.125,0,1}};
        if (make_canvas(17,19,17,19,horizon).ok) throw std::runtime_error("Horizon crossing accepted");
        // Exhaustively cover all 4x4 masks: holes, disconnected regions, flat
        // histograms, increasing/decreasing stacks, and equal-area/width ties.
        for (unsigned bits = 0; bits < (1u << 16); ++bits) {
            std::vector<uint8_t> mask(16);
            for (int i = 0; i < 16; ++i) mask[i] = (bits >> i) & 1u;
            check(mask, 4, 4);
        }
        unsigned seed = 1234567;
        for (int trial = 0; trial < 1000; ++trial) {
            const int w = 1 + trial % 9, h = 1 + (trial / 9) % 8;
            std::vector<uint8_t> mask(w * h);
            for (auto& pixel : mask) {
                seed = seed * 1664525u + 1013904223u;
                pixel = (seed >> 31) & 1u;
            }
            check(mask, w, h);
        }
        // Exercise several column/row blocks and a reduction with >256 rows.
        for (int fill = 0; fill <= 1; ++fill) {
            check(std::vector<uint8_t>(513, fill), 513, 1);
            check(std::vector<uint8_t>(513, fill), 1, 513);
        }
        std::vector<uint8_t> tied_rows(513, 0);
        for (int y = 0; y < 513; y += 2) tied_rows[y] = 1;
        check(tied_rows, 1, 513);
        std::cout << "PASS: canvas checks, 65,536 exhaustive masks, 1,000 random masks, 5 boundary cases\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
