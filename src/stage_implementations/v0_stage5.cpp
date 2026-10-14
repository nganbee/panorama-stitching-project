// File: v0_stage5.cpp
// Authors: Team 8; maintenance assisted by OpenAI Codex.
// Description: V0 (sequential) homography estimation using deterministic
//              RANSAC, 4-point DLT, reprojection-error scoring, and an
//              inlier-only least-squares refinement.
// External sources: none.
#include "../core_headers/common.h"
#include "../core_headers/v0_stages.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <utility>
#include <vector>

namespace {

constexpr double kEpsilon = 1e-12;

struct Point {
    double x = 0.0;
    double y = 0.0;
};

struct Normalization {
    double T[9]{};
    double invT[9]{};
};

bool solve_linear_system(double* a, double* b, int n) {
    for (int col = 0; col < n; ++col) {
        int pivot = col;
        double pivot_abs = std::fabs(a[col * n + col]);
        for (int row = col + 1; row < n; ++row) {
            const double v = std::fabs(a[row * n + col]);
            if (v > pivot_abs) {
                pivot = row;
                pivot_abs = v;
            }
        }
        if (pivot_abs < kEpsilon) return false;

        if (pivot != col) {
            for (int j = col; j < n; ++j)
                std::swap(a[col * n + j], a[pivot * n + j]);
            std::swap(b[col], b[pivot]);
        }

        const double inv_pivot = 1.0 / a[col * n + col];
        for (int j = col; j < n; ++j) a[col * n + j] *= inv_pivot;
        b[col] *= inv_pivot;

        for (int row = 0; row < n; ++row) {
            if (row == col) continue;
            const double factor = a[row * n + col];
            if (std::fabs(factor) < kEpsilon) continue;
            for (int j = col; j < n; ++j)
                a[row * n + j] -= factor * a[col * n + j];
            b[row] -= factor * b[col];
        }
    }
    return true;
}

Normalization make_normalization(const std::vector<Point>& points) {
    Normalization n;
    double cx = 0.0, cy = 0.0;
    for (const auto& p : points) {
        cx += p.x;
        cy += p.y;
    }
    cx /= static_cast<double>(points.size());
    cy /= static_cast<double>(points.size());

    double mean_distance = 0.0;
    for (const auto& p : points) {
        const double dx = p.x - cx;
        const double dy = p.y - cy;
        mean_distance += std::sqrt(dx * dx + dy * dy);
    }
    mean_distance /= static_cast<double>(points.size());
    const double scale = mean_distance > kEpsilon ? std::sqrt(2.0) / mean_distance : 1.0;

    n.T[0] = scale; n.T[1] = 0.0;  n.T[2] = -scale * cx;
    n.T[3] = 0.0;   n.T[4] = scale; n.T[5] = -scale * cy;
    n.T[6] = 0.0;   n.T[7] = 0.0;  n.T[8] = 1.0;

    n.invT[0] = 1.0 / scale; n.invT[1] = 0.0;           n.invT[2] = cx;
    n.invT[3] = 0.0;         n.invT[4] = 1.0 / scale;   n.invT[5] = cy;
    n.invT[6] = 0.0;         n.invT[7] = 0.0;            n.invT[8] = 1.0;
    return n;
}

Point apply_h(const double h[9], const Point& p, bool& valid) {
    const double z = h[6] * p.x + h[7] * p.y + h[8];
    if (std::fabs(z) < kEpsilon) {
        valid = false;
        return {};
    }
    Point mapped{
        (h[0] * p.x + h[1] * p.y + h[2]) / z,
        (h[3] * p.x + h[4] * p.y + h[5]) / z
    };
    valid = std::isfinite(mapped.x) && std::isfinite(mapped.y);
    return mapped;
}

void multiply3x3(const double a[9], const double b[9], double out[9]) {
    double tmp[9]{};
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col)
            for (int k = 0; k < 3; ++k)
                tmp[row * 3 + col] += a[row * 3 + k] * b[k * 3 + col];
    for (int i = 0; i < 9; ++i) out[i] = tmp[i];
}

bool normalize_homography(double h[9]) {
    if (!std::isfinite(h[8]) || std::fabs(h[8]) < kEpsilon) return false;
    for (int i = 0; i < 9; ++i) h[i] /= h[8];
    for (int i = 0; i < 9; ++i) if (!std::isfinite(h[i])) return false;
    return true;
}

bool estimate_homography(const std::vector<Point>& src,
                         const std::vector<Point>& dst,
                         const std::vector<int>& selected,
                         double h[9]) {
    if (selected.size() < 4) return false;

    std::vector<Point> src_selected;
    std::vector<Point> dst_selected;
    src_selected.reserve(selected.size());
    dst_selected.reserve(selected.size());
    for (int index : selected) {
        src_selected.push_back(src[index]);
        dst_selected.push_back(dst[index]);
    }

    const Normalization ns = make_normalization(src_selected);
    const Normalization nd = make_normalization(dst_selected);

    // Solve the normalized 8-parameter DLT system with h[8] = 1.
    double ata[64]{};
    double atb[8]{};
    for (size_t i = 0; i < selected.size(); ++i) {
        const Point s{
            ns.T[0] * src_selected[i].x + ns.T[2],
            ns.T[4] * src_selected[i].y + ns.T[5]
        };
        const Point d{
            nd.T[0] * dst_selected[i].x + nd.T[2],
            nd.T[4] * dst_selected[i].y + nd.T[5]
        };

        const double rows[2][8] = {
            {s.x, s.y, 1.0, 0.0, 0.0, 0.0, -d.x * s.x, -d.x * s.y},
            {0.0, 0.0, 0.0, s.x, s.y, 1.0, -d.y * s.x, -d.y * s.y}
        };
        const double rhs[2] = {d.x, d.y};
        for (int r = 0; r < 2; ++r) {
            for (int c = 0; c < 8; ++c) {
                atb[c] += rows[r][c] * rhs[r];
                for (int dcol = 0; dcol < 8; ++dcol)
                    ata[c * 8 + dcol] += rows[r][c] * rows[r][dcol];
            }
        }
    }

    if (!solve_linear_system(ata, atb, 8)) return false;
    double hn[9] = {atb[0], atb[1], atb[2], atb[3], atb[4], atb[5], atb[6], atb[7], 1.0};

    double tmp[9]{};
    multiply3x3(hn, ns.T, tmp);
    multiply3x3(nd.invT, tmp, h);
    return normalize_homography(h);
}

double squared_reprojection_error(const double h[9], const Point& src, const Point& dst) {
    bool valid = false;
    const Point mapped = apply_h(h, src, valid);
    if (!valid) return std::numeric_limits<double>::infinity();
    const double dx = mapped.x - dst.x;
    const double dy = mapped.y - dst.y;
    return dx * dx + dy * dy;
}

}  // namespace

bool v0_ransac_homography(const std::vector<Keypoint>& kpA,
                          const std::vector<Keypoint>& kpB,
                          const std::vector<Match>& matches,
                          Homography& H,
                          int& nInliers) {
    nInliers = 0;
    H = Homography{};
    if (matches.size() < 4) return false;

    std::vector<Point> src;
    std::vector<Point> dst;
    src.reserve(matches.size());
    dst.reserve(matches.size());
    for (const auto& m : matches) {
        if (m.idxA < 0 || m.idxA >= static_cast<int>(kpA.size()) ||
            m.idxB < 0 || m.idxB >= static_cast<int>(kpB.size())) continue;
        if (!std::isfinite(kpA[m.idxA].x) || !std::isfinite(kpA[m.idxA].y) ||
            !std::isfinite(kpB[m.idxB].x) || !std::isfinite(kpB[m.idxB].y)) continue;
        // H maps image B (source) into image A (destination).
        src.push_back({kpB[m.idxB].x, kpB[m.idxB].y});
        dst.push_back({kpA[m.idxA].x, kpA[m.idxA].y});
    }
    if (src.size() < 4) return false;

    std::mt19937 rng(params::kRansacSeed);
    std::uniform_int_distribution<int> pick(0, static_cast<int>(src.size()) - 1);
    int best_count = -1;
    double best_error = std::numeric_limits<double>::infinity();
    std::vector<int> best_inliers;

    for (int iteration = 0; iteration < params::kRansacIters; ++iteration) {
        std::array<int, 4> sample{};
        int count = 0;
        while (count < 4) {
            const int candidate = pick(rng);
            bool duplicate = false;
            for (int j = 0; j < count; ++j) duplicate = duplicate || sample[j] == candidate;
            if (!duplicate) sample[count++] = candidate;
        }

        std::vector<int> selected(sample.begin(), sample.end());
        double candidate_h[9]{};
        if (!estimate_homography(src, dst, selected, candidate_h)) continue;

        int count_inliers = 0;
        double total_error = 0.0;
        std::vector<int> inliers;
        for (int i = 0; i < static_cast<int>(src.size()); ++i) {
            const double error = squared_reprojection_error(candidate_h, src[i], dst[i]);
            if (error <= static_cast<double>(params::kRansacThresh) * params::kRansacThresh) {
                ++count_inliers;
                total_error += error;
                inliers.push_back(i);
            }
        }

        if (count_inliers > best_count ||
            (count_inliers == best_count && total_error < best_error)) {
            best_count = count_inliers;
            best_error = total_error;
            best_inliers = std::move(inliers);
        }
    }

    if (best_inliers.size() < 4) return false;
    if (!estimate_homography(src, dst, best_inliers, H.h)) return false;
    // Report the consensus of the returned, refined H, not the old hypothesis.
    for (size_t i = 0; i < src.size(); ++i)
        if (squared_reprojection_error(H.h, src[i], dst[i]) <=
            double(params::kRansacThresh) * params::kRansacThresh) ++nInliers;
    return nInliers >= 4;
}
