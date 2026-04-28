
#ifndef MATH_UTILITIES_CUH
#define MATH_UTILITIES_CUH

#include <cstddef>
#include <cmath>
#include <type_traits>
#include <utility>
#include "../constants.cuh"
#include "types.cuh"

// jet shape inlet
__device__ [[nodiscard]] constexpr label_t isJet(label_t x,
                                                 label_t z) noexcept
{
    const real_t dx = static_cast<real_t>(x) - jet_x0;
    const real_t dz = static_cast<real_t>(z) - jet_z0;

    return (dx * dx + dz * dz <= jet_radius * jet_radius) ? 1 : 0;
}

// Extrapolation
__device__ __forceinline__ real_t extrapolation(const real_t phiF,
                                                const real_t phiFF)
{
    // return static_cast<real_t>(2.0) * phiF - phiFF;

    constexpr real_t beta = real_t(0.5);
    return phiF + beta * (phiF - phiFF);
}

// Convective oulet
__device__ __forceinline__ real_t convectiveB(const real_t phiBold,
                                              const real_t phiF,
                                              const real_t uc) noexcept
{
    return phiBold - uc * (phiBold - phiF);
}

// Sponge layer
inline constexpr real_t sponge_gain = static_cast<real_t>(3.0);
inline constexpr real_t sponge_K = static_cast<real_t>(100.0);

inline constexpr label_t sponge_y_end = NY - static_cast<label_t>(2);
inline constexpr label_t sponge_y_start = sponge_y_end - sponge_cells + static_cast<label_t>(1);

__device__ __forceinline__ real_t clamp01(const real_t x) noexcept
{
    return fminf(fmaxf(x, real_t(0.0)), real_t(1.0));
}

__device__ __forceinline__ real_t sponge_s(const label_t y) noexcept
{
    constexpr real_t inv_width = static_cast<real_t>(1.0) / real_t(sponge_y_end - sponge_y_start);

    const real_t yn = static_cast<real_t>(y);
    const real_t s = (yn - static_cast<real_t>(sponge_y_start)) * inv_width;

    return clamp01(s);
}

__device__ __forceinline__ real_t sponge_pow(const real_t s) noexcept
{
    // p = 3
    return s * s * s;
}

__device__ __forceinline__ real_t nu_sponge(const label_t y) noexcept
{
    const real_t s = sponge_s(y);
    return nu * (real_t(1.0) + sponge_K * sponge_pow(s));
}

__device__ __forceinline__ real_t tau_sponge(const label_t y) noexcept
{
    return real_t(0.5) + static_cast<real_t>((nu_sponge(y)) / cs2);
}

__device__ __forceinline__ real_t omega_sponge(const label_t y) noexcept
{
    return static_cast<real_t>(static_cast<real_t>(1.0) / tau_sponge(y));
}

#endif
