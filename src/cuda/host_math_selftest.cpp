// Authors: Team 8, with Codex implementation assistance. External sources: none.
// Checks only the host/device mathematical utility on the CPU. This is not
// evidence of CUDA compilation, numerical parity or GPU performance.
#include "homography_math.cuh"
#include <iostream>
#include <vector>

int main() {
    using cuda_homography::Point;
    const Homography expected{{1.02,0.03,7,-0.01,0.98,3,0.0003,-0.0001,1}};
    std::vector<Point> src,dst;
    std::vector<int> selected;
    for (int y=0;y<6;++y) for (int x=0;x<8;++x) {
        const Point p{10.0+x*9,8.0+y*10};
        const double z=expected.h[6]*p.x+expected.h[7]*p.y+expected.h[8];
        selected.push_back(static_cast<int>(src.size())); src.push_back(p);
        dst.push_back({(expected.h[0]*p.x+expected.h[1]*p.y+expected.h[2])/z,
                       (expected.h[3]*p.x+expected.h[4]*p.y+expected.h[5])/z});
    }
    for (const std::vector<int> subset : {std::vector<int>{0,7,40,47},selected}) {
        Homography actual{};
        if (!cuda_homography::estimate(src.data(),dst.data(),subset.data(),static_cast<int>(subset.size()),actual)) return 1;
        for (int i=0;i<9;++i) if (std::fabs(actual.h[i]-expected.h[i])>1e-8) return 2;
        for (std::size_t i=0;i<src.size();++i)
            if (cuda_homography::squared_error(actual,src[i],dst[i])>1e-14) return 3;
    }
    const Point degenerate[4]{{1,1},{1,1},{1,1},{1,1}};
    const int indices[4]{0,1,2,3}; Homography invalid{};
    if (cuda_homography::estimate(degenerate,degenerate,indices,4,invalid)) return 4;
    Homography horizon{{1,0,0,0,1,0,1,0,-1}};
    if (cuda_homography::squared_error(horizon,{1,2},{1,2})!=DBL_MAX) return 5;
    std::cout<<"Host homography utility PASS (GPU execution still pending)\n";
    return 0;
}
