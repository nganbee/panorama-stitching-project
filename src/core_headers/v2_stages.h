// External sources: none; extends the team's V1 matching implementation.
#pragma once
#include "common.h"

// V2 changes only S4 nearest-neighbour search to shared-memory tiling.
// Implemented independently in v2_stage4.cu; V1 remains unchanged.
void v2_match(const std::vector<Descriptor> &a, const std::vector<Descriptor> &b,
              std::vector<Match> &matches);
