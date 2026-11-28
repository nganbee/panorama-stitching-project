#pragma once
#include "common.h"

// Device-resident S4: same contract as v5::match, one warp per query, direct global candidate reads.
namespace v6 {
int match(const Descriptor* a, int na, const Descriptor* b, int nb, Match* matches);
}
