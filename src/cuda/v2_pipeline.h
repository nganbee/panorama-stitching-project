// Authors: Nguyen Duy Khanh; implementation assisted by OpenAI Codex.
// External sources: none; inherits the team's V1 pipeline interface.
#pragma once
#include "../core_headers/common.h"

// V2 inherits V1 and replaces only S4 with shared-memory descriptor tiling.
bool run_v2(const Image& A, const Image& B, PipelineResult& out);

namespace pipeline_detail {
using Stage6Fn = void (*)(const Image&, const Image&, const Homography&, Image&);
using BlurFn = void (*)(const Gray&, Gray&);

// V2/V3 share S1–S5; V4 can substitute blur while retaining V2 matching.
// A null blur selects the original V1 Gaussian blur used by V2/V3.
bool run_v2_with_stage6(const Image& A, const Image& B, PipelineResult& out,
                       Stage6Fn stage6, const char* version_name, BlurFn blur = nullptr);
}
