// Authors: Nguyen Duy Khanh; implementation assisted by OpenAI Codex.
// External sources: none; inherits the team's V1 pipeline interface.
#pragma once
#include "../core_headers/common.h"

// V2 inherits V1 and replaces only S4 with shared-memory descriptor tiling.
bool run_v2(const Image& A, const Image& B, PipelineResult& out);
