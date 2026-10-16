// File: run_stage12_selftest.cpp
// Authors: Quoc Thinh
// Description: Small executable wrapper for the independent S1-S2 self-tests.
// External sources: none.
#include "../src/core_headers/v0_stages.h"

int main() {
    return v0_selftest_stage12() ? 0 : 1;
}
