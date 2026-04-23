#ifndef BOUNDS_CUH
#define BOUNDS_CUH

#include "../constants.cuh"

__device__ [[nodiscard]] constexpr inline bool interior(const label_t x, const label_t y, const label_t z) noexcept
{

    return (x >= NX || y >= NY || z >= NZ ||
            x == 0 || x == NX - 1 ||
            y == 0 || y == NY - 1 ||
            z == 0 || z == NZ - 1);
}

__device__ [[nodiscard]] constexpr inline bool inlet_outlet_interior(const label_t x, const label_t z) noexcept
{
    return (x >= NX || z >= NZ ||
            x == 0 || x == NX - 1 ||
            z == 0 || z == NZ - 1);
}

//  Periodic boundary conditions
__device__ [[nodiscard]] __forceinline__ constexpr label_t wrapx(const label_t x) noexcept
{
    if (x == 0)
    {
        return NX - 2;
    }
    if (x == NX - 1)
    {
        return 1;
    }
    return x;
}

__device__ [[nodiscard]] __forceinline__ constexpr label_t wrapy(const label_t y) noexcept
{
    if (y == 0)
    {
        return NY - 2;
    }
    if (y == NY - 1)
    {
        return 1;
    }
    return y;
}

__device__ [[nodiscard]] __forceinline__ constexpr label_t wrapz(const label_t z) noexcept
{
    if (z == 0)
    {
        return NZ - 2;
    }
    if (z == NZ - 1)
    {
        return 1;
    }
    return z;
}

#endif