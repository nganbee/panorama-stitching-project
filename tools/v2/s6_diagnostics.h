// Authors: Team 8; V2 verification assisted by OpenAI Codex.
// External sources: none. Test-only observations; excluded from timed pipelines.
#pragma once
#include "../../src/core_headers/common.h"

struct V2S6Diagnostics {
    bool ok = false;
    int origin_x = 0, origin_y = 0, width = 0, height = 0;
    int crop_x = 0, crop_y = 0, crop_w = 0, crop_h = 0;
    std::vector<uint8_t> mask;
};
V2S6Diagnostics v2_cpu_s6_diagnostics(const Image&, const Image&, const Homography&);
V2S6Diagnostics v2_gpu_s6_diagnostics(const Image&, const Image&, const Homography&);
