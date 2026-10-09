// File: v0_pipeline.cpp
// Description: V0 (sequential) pipeline glue. Calls the stage functions declared in v0_stages.h
//              in order and fills StageTimes with per-stage CPU timings.
//              Timings of Stage 1-3 are the SUM over image A and image B.
//              Image loading/saving is NOT part of this function (excluded from timing).
// External sources: none
#include "common.h"
#include "versions.h"
#include "v0_stages.h"
#include <chrono>

namespace {

using Clock = std::chrono::high_resolution_clock;

inline double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

struct Features {
    std::vector<Keypoint> kps;
    std::vector<Descriptor> desc;
};

// Stages 1-3 on one image. Scratch buffers are static so that allocation happens in the warm-up run.
void extract_features(const Image& img, Features& f, StageTimes& t) {
    static Gray gray, blurred, resp;

    auto s = Clock::now();
    v0_rgb_to_gray(img, gray);
    t.gray += ms_since(s);

    s = Clock::now();
    v0_gaussian_blur(gray, blurred);
    t.blur += ms_since(s);

    s = Clock::now();
    v0_harris_response(blurred, resp);
    t.harris += ms_since(s);

    s = Clock::now();
    v0_nms_topk(resp, f.kps);
    t.nms_topk += ms_since(s);

    s = Clock::now();
    v0_describe(blurred, f.kps, f.desc);      // S3 uses the blurred image of THIS image
    t.describe += ms_since(s);
}

}  // namespace

bool run_v0(const Image& A, const Image& B, PipelineResult& out) {
    out = PipelineResult{};
    StageTimes& t = out.times;
    const auto t_total = Clock::now();

    Features fa, fb;
    extract_features(A, fa, t);
    extract_features(B, fb, t);
    out.nKpA = (int)fa.kps.size();
    out.nKpB = (int)fb.kps.size();

    std::vector<Match> matches;
    auto s = Clock::now();
    v0_match(fa.desc, fb.desc, matches);
    t.match = ms_since(s);
    out.nMatches = (int)matches.size();
    if (matches.size() < 4) { t.end_to_end = ms_since(t_total); return false; }   // DLT needs >= 4

    s = Clock::now();
    bool ok = v0_ransac_homography(fa.kps, fb.kps, matches, out.H, out.nInliers);
    t.ransac = ms_since(s);
    if (!ok) { t.end_to_end = ms_since(t_total); return false; }

    s = Clock::now();
    v0_warp_blend(A, B, out.H, out.panorama);
    t.warp_blend = ms_since(s);

    t.end_to_end = ms_since(t_total);
    return true;
}
