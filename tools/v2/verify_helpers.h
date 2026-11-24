// Authors: Team 8; V2 verification assisted by OpenAI Codex.
// External sources: none. Acceptance criteria from docs/PLAN.md.
#pragma once
#include "../../src/core_headers/common.h"
#include <algorithm>
#include <cmath>
#include <limits>

inline bool v2_equal_matches(const std::vector<Match>& a, const std::vector<Match>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i].idxA != b[i].idxA || a[i].idxB != b[i].idxB || a[i].dist != b[i].dist) return false;
    return true;
}
inline bool v2_equal_keypoints(const std::vector<Keypoint>& a, const std::vector<Keypoint>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i].x != b[i].x || a[i].y != b[i].y ||
            !std::isfinite(a[i].score) || !std::isfinite(b[i].score) ||
            std::fabs(a[i].score-b[i].score) > 5e-2 + 2e-5 * std::fabs(a[i].score)) return false;
    return true;
}
inline bool v2_project(const Homography& h, double x, double y, double& u, double& v) {
    const double z = h.h[6]*x + h.h[7]*y + h.h[8];
    if (!std::isfinite(z) || std::fabs(z) < 1e-12) return false;
    u = (h.h[0]*x+h.h[1]*y+h.h[2])/z; v = (h.h[3]*x+h.h[4]*y+h.h[5])/z;
    return std::isfinite(u) && std::isfinite(v);
}
inline double v2_projection_error(const Homography& a, const Homography& b, int w, int h,
                                  const std::vector<Keypoint>& points) {
    double worst = 0;
    auto test = [&](double x, double y) {
        double ax, ay, bx, by;
        if (!v2_project(a,x,y,ax,ay) || !v2_project(b,x,y,bx,by)) {
            worst = std::numeric_limits<double>::infinity(); return;
        }
        worst = std::max(worst, std::hypot(ax-bx, ay-by));
    };
    for (int y = 0; y <= 10; ++y) for (int x = 0; x <= 10; ++x)
        test((w-1.0)*x/10, (h-1.0)*y/10);
    for (const auto& p : points) test(p.x,p.y);
    return worst;
}
