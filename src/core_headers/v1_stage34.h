#pragma once
#include "common.h"

// Blocking host-buffer interfaces for the baseline global-memory CUDA stages.
// Invalid S3 input returns an empty output, matching V0. CUDA failures throw
// std::runtime_error; these functions never fall back to CPU computation.
void v1_describe(const Gray& blurred, const std::vector<Keypoint>& kps,
                 std::vector<Descriptor>& desc);
void v1_match(const std::vector<Descriptor>& a, const std::vector<Descriptor>& b,
              std::vector<Match>& matches);
