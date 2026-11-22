// Authors: Team 8; V2 verification assisted by OpenAI Codex.
// External sources: none. Acceptance criteria from docs/PLAN.md.
#pragma once
#include "s6_diagnostics.h"
#include <algorithm>
#include <cmath>
#include <limits>

inline bool v2_equal_matches(const std::vector<Match>& a, const std::vector<Match>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i].idxA != b[i].idxA || a[i].idxB != b[i].idxB || a[i].dist != b[i].dist) return false;
    return true;
}
inline bool v2_equal_descriptors(const std::vector<Descriptor>& a, const std::vector<Descriptor>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        for (int k = 0; k < 4; ++k) if (a[i].bits[k] != b[i].bits[k]) return false;
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
inline bool v2_equal_canvas(const V2S6Diagnostics& a, const V2S6Diagnostics& b) {
    return a.ok && b.ok && a.origin_x==b.origin_x && a.origin_y==b.origin_y &&
           a.width==b.width && a.height==b.height;
}
inline bool v2_equal_crop(const V2S6Diagnostics& a, const V2S6Diagnostics& b) {
    return a.crop_x==b.crop_x && a.crop_y==b.crop_y && a.crop_w==b.crop_w && a.crop_h==b.crop_h;
}
