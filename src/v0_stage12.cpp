// Gray + Blur + Harris + NMS + top-K
// File: v0_stage12.cpp
// Authors: Quoc Thinh
// Description: V0 (sequential) Stage 1 (grayscale + separable Gaussian blur)
//              and Stage 2 (Harris response, NMS, top-K).
// External sources: none
#include "common.h"
#include "v0_stages.h"
#include <algorithm>
#include <cmath>

static inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

// Stage 1
void v0_rgb_to_gray(const Image& in, Gray& gray) {
    gray.w = in.w; gray.h = in.h;
    gray.data.resize((size_t)in.w * in.h);
    const uint8_t* p = in.data.data();
    for (size_t i = 0; i < gray.data.size(); ++i)
        gray.data[i] = 0.299f * p[3 * i] + 0.587f * p[3 * i + 1] + 0.114f * p[3 * i + 2];
}

static void make_gaussian_kernel(float* k, float sigma, int r) {
    float sum = 0.f;
    for (int i = -r; i <= r; ++i) {
        k[i + r] = std::exp(-(float)(i * i) / (2.f * sigma * sigma));
        sum += k[i + r];
    }
    for (int i = 0; i < 2 * r + 1; ++i) k[i] /= sum;
}

void v0_gaussian_blur(const Gray& in, Gray& out) {
    const int W = in.w, H = in.h, r = params::kGaussRadius;
    float k[2 * params::kGaussRadius + 1];
    make_gaussian_kernel(k, params::kGaussSigma, r);

    out.w = W; out.h = H;
    out.data.resize((size_t)W * H);
    static std::vector<float> tmp;          // reused across runs; first alloc happens in warm-up
    tmp.resize((size_t)W * H);

    // horizontal pass: in -> tmp
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            float acc = 0.f;
            for (int d = -r; d <= r; ++d)
                acc += k[d + r] * in.data[(size_t)y * W + clampi(x + d, 0, W - 1)];
            tmp[(size_t)y * W + x] = acc;
        }
    // vertical pass: tmp -> out
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            float acc = 0.f;
            for (int d = -r; d <= r; ++d)
                acc += k[d + r] * tmp[(size_t)clampi(y + d, 0, H - 1) * W + x];
            out.data[(size_t)y * W + x] = acc;
        }
}

// Stage 2
void v0_harris_response(const Gray& g, Gray& R) {
    const int W = g.w, H = g.h, wr = params::kHarrisWinRadius;
    const float* I = g.data.data();
    auto at = [&](int x, int y) { return I[(size_t)clampi(y, 0, H - 1) * W + clampi(x, 0, W - 1)]; };

    static std::vector<float> Ixx, Iyy, Ixy;
    Ixx.resize((size_t)W * H); Iyy.resize((size_t)W * H); Ixy.resize((size_t)W * H);

    // Sobel gradients (scaled by 1/8) and tensor products
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            float ix = ((at(x+1,y-1) + 2*at(x+1,y) + at(x+1,y+1)) -
                        (at(x-1,y-1) + 2*at(x-1,y) + at(x-1,y+1))) * 0.125f;
            float iy = ((at(x-1,y+1) + 2*at(x,y+1) + at(x+1,y+1)) -
                        (at(x-1,y-1) + 2*at(x,y-1) + at(x+1,y-1))) * 0.125f;
            size_t i = (size_t)y * W + x;
            Ixx[i] = ix * ix; Iyy[i] = iy * iy; Ixy[i] = ix * iy;
        }

    // window sum + Harris response R = det(M) - k * trace(M)^2
    R.w = W; R.h = H;
    R.data.assign((size_t)W * H, 0.f);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            float sxx = 0, syy = 0, sxy = 0;
            for (int dy = -wr; dy <= wr; ++dy) {
                int yy = clampi(y + dy, 0, H - 1);
                for (int dx = -wr; dx <= wr; ++dx) {
                    size_t j = (size_t)yy * W + clampi(x + dx, 0, W - 1);
                    sxx += Ixx[j]; syy += Iyy[j]; sxy += Ixy[j];
                }
            }
            float det = sxx * syy - sxy * sxy, tr = sxx + syy;
            R.data[(size_t)y * W + x] = det - params::kHarrisK * tr * tr;
        }
}

void v0_nms_topk(const Gray& R, std::vector<Keypoint>& kps) {
    const int W = R.w, H = R.h, nr = params::kNmsRadius, b = params::kBorder;
    const float* r = R.data.data();

    float rmax = 0.f;
    for (size_t i = 0; i < R.data.size(); ++i) rmax = std::max(rmax, r[i]);
    const float thr = params::kHarrisThresh * rmax;

    kps.clear();
    for (int y = b; y < H - b; ++y)
        for (int x = b; x < W - b; ++x) {
            size_t i = (size_t)y * W + x;
            float v = r[i];
            if (v <= thr) continue;
            bool is_max = true;
            for (int dy = -nr; dy <= nr && is_max; ++dy)
                for (int dx = -nr; dx <= nr; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    int ny = y + dy, nx = x + dx;
                    if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
                    size_t j = (size_t)ny * W + nx;
                    // strict maximum; ties broken by raster index so result is deterministic
                    if (r[j] > v || (r[j] == v && j < i)) { is_max = false; break; }
                }
            if (is_max) kps.push_back({(float)x, (float)y, v});
        }

    auto better = [](const Keypoint& a, const Keypoint& b2) {
        if (a.score != b2.score) return a.score > b2.score;
        if (a.y != b2.y) return a.y < b2.y;
        return a.x < b2.x;
    };
    if ((int)kps.size() > params::kTopK) {
        std::partial_sort(kps.begin(), kps.begin() + params::kTopK, kps.end(), better);
        kps.resize(params::kTopK);
    } else {
        std::sort(kps.begin(), kps.end(), better);
    }
}