// Fixture generator for the ptxas-verbose backend: three kernels with distinct
// codegen character. Compile: nvcc -O3 -arch=sm_89 -Xptxas -v -c ptxas_fixture.cu
#include <cuda_runtime.h>

// lean: few registers, no spills, no smem
__global__ void lean_saxpy(float a, const float* __restrict__ x,
                           float* __restrict__ y, long n) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

// spilly: a large dynamically-indexed local array forces stack + spill traffic
__global__ void spilly_kernel(const float* __restrict__ x, float* __restrict__ y,
                              int n, int rot) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float buf[96];
    #pragma unroll
    for (int k = 0; k < 96; ++k) buf[k] = x[(i + k) % n];
    float acc = 0.f;
    #pragma unroll
    for (int k = 0; k < 96; ++k) acc += buf[(k + rot) % 96] * buf[(k * 7 + rot) % 96];
    y[i] = acc;
}

// smem_tile: static shared memory tile
__global__ void smem_tile_kernel(const float* __restrict__ x,
                                 float* __restrict__ y, int n) {
    __shared__ float tile[32][33];  // 33 to avoid bank conflicts
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    tile[threadIdx.y][threadIdx.x] = (i < n) ? x[i] : 0.f;
    __syncthreads();
    if (i < n) y[i] = tile[threadIdx.x % 32][threadIdx.y % 32] * 2.f;
}
// many live scalar accumulators + register cap => genuine spill stores/loads
__global__ void true_spill_kernel(const float* __restrict__ x, float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float a00=x[i],a01=x[i+1],a02=x[i+2],a03=x[i+3],a04=x[i+4],a05=x[i+5],a06=x[i+6],a07=x[i+7];
    float a08=x[i+8],a09=x[i+9],a10=x[i+10],a11=x[i+11],a12=x[i+12],a13=x[i+13],a14=x[i+14],a15=x[i+15];
    float a16=x[i+16],a17=x[i+17],a18=x[i+18],a19=x[i+19],a20=x[i+20],a21=x[i+21],a22=x[i+22],a23=x[i+23];
    float a24=x[i+24],a25=x[i+25],a26=x[i+26],a27=x[i+27],a28=x[i+28],a29=x[i+29],a30=x[i+30],a31=x[i+31];
    for (int k = 0; k < n; ++k) {
        float v = x[(i + k) & 1023];
        a00=fmaf(a00,v,1.f); a01=fmaf(a01,v,2.f); a02=fmaf(a02,v,3.f); a03=fmaf(a03,v,4.f);
        a04=fmaf(a04,v,1.f); a05=fmaf(a05,v,2.f); a06=fmaf(a06,v,3.f); a07=fmaf(a07,v,4.f);
        a08=fmaf(a08,v,1.f); a09=fmaf(a09,v,2.f); a10=fmaf(a10,v,3.f); a11=fmaf(a11,v,4.f);
        a12=fmaf(a12,v,1.f); a13=fmaf(a13,v,2.f); a14=fmaf(a14,v,3.f); a15=fmaf(a15,v,4.f);
        a16=fmaf(a16,v,1.f); a17=fmaf(a17,v,2.f); a18=fmaf(a18,v,3.f); a19=fmaf(a19,v,4.f);
        a20=fmaf(a20,v,1.f); a21=fmaf(a21,v,2.f); a22=fmaf(a22,v,3.f); a23=fmaf(a23,v,4.f);
        a24=fmaf(a24,v,1.f); a25=fmaf(a25,v,2.f); a26=fmaf(a26,v,3.f); a27=fmaf(a27,v,4.f);
        a28=fmaf(a28,v,1.f); a29=fmaf(a29,v,2.f); a30=fmaf(a30,v,3.f); a31=fmaf(a31,v,4.f);
    }
    y[i] = a00+a01+a02+a03+a04+a05+a06+a07+a08+a09+a10+a11+a12+a13+a14+a15
          +a16+a17+a18+a19+a20+a21+a22+a23+a24+a25+a26+a27+a28+a29+a30+a31;
}
