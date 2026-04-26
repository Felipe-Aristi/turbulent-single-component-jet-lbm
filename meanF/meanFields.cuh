#ifndef MEANFIELDS_CUH
#define MEANFIELDS_CUH

#include "../utilities/types.cuh"
#include "../utilities/indexing.cuh"
#include "../constants.cuh"

//------------- Postprocessing -------------------------------

__host__ __device__ [[nodiscard]] __forceinline__ real_t tke(const label_t x, const label_t y, const label_t z,
                                                             const real_t *ux, const real_t *uy, const real_t *uz) noexcept
{
    const label_t id = idx(x, y, z);
    const real_t vx = ux[id];
    const real_t vy = uy[id];
    const real_t vz = uz[id];

    return static_cast<real_t>(0.5) * (vx * vx + vy * vy + vz * vz);
}

__host__ __device__ [[nodiscard]] __forceinline__ real_t running_average(const label_t x, const label_t y, const label_t z,
                                                                         const real_t old_avg,
                                                                         const real_t new_value,
                                                                         const unsigned int n_prev_samples) noexcept
{
    const real_t n = static_cast<real_t>(n_prev_samples);
    return (old_avg * n + new_value) / (n + static_cast<real_t>(1));
}

__host__ __device__ [[nodiscard]] __forceinline__ std::size_t radial_profile_idx(const label_t y,
                                                                                 const label_t rbin) noexcept
{
    return static_cast<std::size_t>(y) * static_cast<std::size_t>(NR_BINS) + static_cast<std::size_t>(rbin);
}

__device__ [[nodiscard]] __forceinline__ real_t radial_velocity(const label_t x,
                                                                const label_t z,
                                                                const real_t ux,
                                                                const real_t uz,
                                                                const real_t r) noexcept
{
    if (r <= static_cast<real_t>(0))
    {
        return static_cast<real_t>(0);
    }

    const real_t dx = static_cast<real_t>(x) - jet_x0;
    const real_t dz = static_cast<real_t>(z) - jet_z0;

    return (dx * ux + dz * uz) / r;
}

#endif
