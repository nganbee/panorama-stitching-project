#pragma once
#include "../../core_headers/v5_stages.h"
#include "../../core_headers/v6_stages.h"
#include "../../core_headers/cuda_utils.cuh"

// Test-only adapter. Production pipelines keep all descriptors on device.
inline void match_host(const std::vector<Descriptor>& a,const std::vector<Descriptor>& b,
                       std::vector<Match>& out,bool warp) {
    out.clear();
    v1_cuda::Buffer<Descriptor> da(a.size()),db(b.size());
    v1_cuda::Buffer<Match> dm(a.size());
    if (!a.empty()) v1_cuda::check(cudaMemcpy(da.data,a.data(),a.size()*sizeof(Descriptor),cudaMemcpyHostToDevice));
    if (!b.empty()) v1_cuda::check(cudaMemcpy(db.data,b.data(),b.size()*sizeof(Descriptor),cudaMemcpyHostToDevice));
    const int n = warp ? v6::match(da.data,int(a.size()),db.data,int(b.size()),dm.data)
                       : v5::match(da.data,int(a.size()),db.data,int(b.size()),dm.data);
    out.resize(n);
    if (n) v1_cuda::check(cudaMemcpy(out.data(),dm.data,n*sizeof(Match),cudaMemcpyDeviceToHost));
}
inline void match5(const std::vector<Descriptor>& a,const std::vector<Descriptor>& b,std::vector<Match>& out) { match_host(a,b,out,false); }
inline void match6(const std::vector<Descriptor>& a,const std::vector<Descriptor>& b,std::vector<Match>& out) { match_host(a,b,out,true); }
