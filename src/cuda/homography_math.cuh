// Authors: Team 8, with Codex implementation assistance. External sources: none.
// Same normalized 8-parameter least-squares system as V0. No allocations in
// the solver, so a hypothesis can run wholly in one CUDA thread.
#pragma once
#include <cfloat>
#include <cmath>
#include "../core_headers/common.h"

#ifdef __CUDACC__
#define PANORAMA_HD __host__ __device__
#else
#define PANORAMA_HD
#endif

namespace cuda_homography {
struct Point { double x, y; };
constexpr double epsilon = 1e-12;

PANORAMA_HD inline bool finite(double v) {
    return v == v && fabs(v) <= DBL_MAX;
}

PANORAMA_HD inline void multiply(const double* a, const double* b, double* out) {
    double tmp[9]{};
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            for (int k = 0; k < 3; ++k) tmp[r * 3 + c] += a[r * 3 + k] * b[k * 3 + c];
    for (int i = 0; i < 9; ++i) out[i] = tmp[i];
}

PANORAMA_HD inline bool solve(double* a, double* b) {
    for (int c = 0; c < 8; ++c) {
        int pivot = c;
        double value = fabs(a[c * 8 + c]);
        for (int r = c + 1; r < 8; ++r) {
            const double next = fabs(a[r * 8 + c]);
            if (next > value) { pivot = r; value = next; }
        }
        if (!finite(value) || value < epsilon) return false;
        if (pivot != c) {
            for (int j = c; j < 8; ++j) {
                const double temp = a[c * 8 + j];
                a[c * 8 + j] = a[pivot * 8 + j]; a[pivot * 8 + j] = temp;
            }
            const double temp = b[c]; b[c] = b[pivot]; b[pivot] = temp;
        }
        const double inv = 1.0 / a[c * 8 + c];
        for (int j = c; j < 8; ++j) a[c * 8 + j] *= inv;
        b[c] *= inv;
        for (int r = 0; r < 8; ++r) {
            if (r == c) continue;
            const double factor = a[r * 8 + c];
            if (fabs(factor) < epsilon) continue;
            for (int j = c; j < 8; ++j) a[r * 8 + j] -= factor * a[c * 8 + j];
            b[r] -= factor * b[c];
        }
    }
    return true;
}

PANORAMA_HD inline void normalization(const Point* p, const int* selected,
                                              int n, double* t, double* inv) {
    double cx = 0, cy = 0;
    for (int i = 0; i < n; ++i) { cx += p[selected[i]].x; cy += p[selected[i]].y; }
    cx /= n; cy /= n;
    double distance = 0;
    for (int i = 0; i < n; ++i) {
        const double x = p[selected[i]].x - cx, y = p[selected[i]].y - cy;
        distance += sqrt(x * x + y * y);
    }
    distance /= n;
    const double scale = distance > epsilon ? sqrt(2.0) / distance : 1.0;
    for (int i = 0; i < 9; ++i) t[i] = inv[i] = 0;
    t[0] = t[4] = scale; t[2] = -scale * cx; t[5] = -scale * cy; t[8] = 1;
    inv[0] = inv[4] = 1.0 / scale; inv[2] = cx; inv[5] = cy; inv[8] = 1;
}

PANORAMA_HD inline bool estimate(const Point* src, const Point* dst,
                                        const int* selected, int n, Homography& out) {
    if (n < 4) return false;
    double ts[9], invs[9], td[9], invd[9];
    normalization(src, selected, n, ts, invs);
    normalization(dst, selected, n, td, invd);
    double ata[64]{}, atb[8]{};
    for (int i = 0; i < n; ++i) {
        const Point s{ts[0] * src[selected[i]].x + ts[2], ts[4] * src[selected[i]].y + ts[5]};
        const Point d{td[0] * dst[selected[i]].x + td[2], td[4] * dst[selected[i]].y + td[5]};
        const double rows[2][8] = {
            {s.x, s.y, 1, 0, 0, 0, -d.x * s.x, -d.x * s.y},
            {0, 0, 0, s.x, s.y, 1, -d.y * s.x, -d.y * s.y}
        };
        const double rhs[2]{d.x, d.y};
        for (int r = 0; r < 2; ++r)
            for (int c = 0; c < 8; ++c) {
                atb[c] += rows[r][c] * rhs[r];
                for (int j = 0; j < 8; ++j) ata[c * 8 + j] += rows[r][c] * rows[r][j];
            }
    }
    if (!solve(ata, atb)) return false;
    const double hn[9]{atb[0], atb[1], atb[2], atb[3], atb[4], atb[5], atb[6], atb[7], 1};
    double tmp[9];
    multiply(hn, ts, tmp); multiply(invd, tmp, out.h);
    if (!finite(out.h[8]) || fabs(out.h[8]) < epsilon) return false;
    const double divisor = out.h[8];
    for (int i = 0; i < 9; ++i) { out.h[i] /= divisor; if (!finite(out.h[i])) return false; }
    return true;
}

PANORAMA_HD inline double squared_error(const Homography& h, Point s, Point d) {
    const double z = h.h[6] * s.x + h.h[7] * s.y + h.h[8];
    if (!finite(z) || fabs(z) < epsilon) return DBL_MAX;
    const double x = (h.h[0] * s.x + h.h[1] * s.y + h.h[2]) / z - d.x;
    const double y = (h.h[3] * s.x + h.h[4] * s.y + h.h[5]) / z - d.y;
    const double value = x * x + y * y;
    return finite(value) ? value : DBL_MAX;
}
} // namespace cuda_homography
#undef PANORAMA_HD
