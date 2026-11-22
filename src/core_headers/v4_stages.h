// Authors: Team 8; implementation assisted by OpenAI Codex.
// V4 changes only Gaussian pixel reads: shared-memory tiles with clamp halos.
#pragma once
#include "common.h"

void v4_gaussian_blur(const Gray& in, Gray& out);
