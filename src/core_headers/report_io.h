// File: report_io.h
// Authors: Team 8; implementation assisted by OpenAI Codex.
// Description: Machine-readable parameters and measured pipeline results.
// External sources: none.
#pragma once
#include "versions.h"
#include <iomanip>
#include <ostream>
#include <string>

inline std::string json_string(const std::string& value) {
    std::string out = "\"";
    const char* digits = "0123456789abcdef";
    for (unsigned char c : value) {
        if (c == '\\' || c == '"') { out += '\\'; out += static_cast<char>(c); }
        else if (c < 32) {
            out += "\\u00"; out += digits[c >> 4]; out += digits[c & 15];
        } else out += static_cast<char>(c);
    }
    return out + '"';
}

inline void write_parameters_json(std::ostream& o) {
    o << std::setprecision(10)
      << "{\"schema_version\":1,\"kGaussSigma\":" << params::kGaussSigma
      << ",\"kGaussRadius\":" << params::kGaussRadius
      << ",\"kHarrisK\":" << params::kHarrisK
      << ",\"kHarrisThresh\":" << params::kHarrisThresh
      << ",\"kHarrisWinRadius\":" << params::kHarrisWinRadius
      << ",\"kNmsRadius\":" << params::kNmsRadius
      << ",\"kTopK\":" << params::kTopK
      << ",\"kBorder\":" << params::kBorder
      << ",\"kPatchSize\":" << params::kPatchSize
      << ",\"kBriefSeed\":" << params::kBriefSeed
      << ",\"kRatio\":" << params::kRatio
      << ",\"kRansacIters\":" << params::kRansacIters
      << ",\"kRansacThresh\":" << params::kRansacThresh
      << ",\"kRansacSeed\":" << params::kRansacSeed
      << ",\"available_versions\":[0";
#ifdef PANORAMA_WITH_CUDA
    o << ",1,2,3";
#endif
    o << "]}\n";
}

inline void write_metrics_json(std::ostream& o, int version, int runs,
                               const Image& A, const Image& B,
                               const PipelineResult& r, const StageTimes& t,
                               const std::string& panorama_path) {
    o << std::setprecision(17)
      << "{\"schema_version\":1,\"status\":\"ok\",\"version\":" << version
      << ",\"runs\":" << runs << ",\"warmup_runs\":1"
      << ",\"imageA\":{\"width\":" << A.w << ",\"height\":" << A.h << '}'
      << ",\"imageB\":{\"width\":" << B.w << ",\"height\":" << B.h << '}'
      << ",\"panorama\":{\"width\":" << r.panorama.w << ",\"height\":" << r.panorama.h
      << ",\"path\":" << json_string(panorama_path) << '}'
      << ",\"features\":{\"keypoints_A\":" << r.nKpA << ",\"keypoints_B\":" << r.nKpB
      << ",\"matches\":" << r.nMatches << ",\"inliers\":" << r.nInliers
      << ",\"inlier_ratio\":" << (r.nMatches ? double(r.nInliers)/r.nMatches : 0.0) << '}'
      << ",\"homography_B_to_A\":[";
    for (int i=0; i<9; ++i) { if (i) o << ','; o << r.H.h[i]; }
    o << "],\"timings_ms\":{\"gray\":" << t.gray << ",\"blur\":" << t.blur
      << ",\"harris\":" << t.harris << ",\"nms_topk\":" << t.nms_topk
      << ",\"describe\":" << t.describe << ",\"match\":" << t.match
      << ",\"ransac\":" << t.ransac << ",\"warp_blend\":" << t.warp_blend
      << ",\"h2d\":" << t.h2d << ",\"d2h\":" << t.d2h
      << ",\"kernel_total\":" << t.kernel_total << ",\"end_to_end\":" << t.end_to_end << '}'
      << ",\"stage_timings_ms\":{\"S1\":" << t.gray+t.blur
      << ",\"S2\":" << t.harris+t.nms_topk << ",\"S3\":" << t.describe
      << ",\"S4\":" << t.match << ",\"S5\":" << t.ransac
      << ",\"S6\":" << t.warp_blend << "}}\n";
}
