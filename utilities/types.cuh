#ifndef TYPES_CUH
#define TYPES_CUH

#include <cstdint>
#include <cuda_fp16.h>

using real_t = float;
using label_t = uint32_t;
using pop_t = __half;

inline constexpr real_t FP16S_UP = real_t(32768.0);         // 2^15
inline constexpr real_t FP16S_DOWN = real_t(1.0 / 32768.0); // 2^-15

__host__ __device__ [[nodiscard]] __forceinline__
    pop_t
    save_pop(const real_t x) noexcept
{
    return __float2half_rn(x * FP16S_UP);
}

__host__ __device__ [[nodiscard]] __forceinline__
    real_t
    load_pop(const pop_t h) noexcept
{
    return __half2float(h) * FP16S_DOWN;
}

#endif