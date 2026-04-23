#ifndef INITIALIZATION_CUH
#define INITIALIZATION_CUH

#include "../utilities/bounds.cuh"
#include "../utilities/mathUtilities.cuh"
#include "../utilities/indexing.cuh"
#include "../utilities/types.cuh"
#include "../constants.cuh"
#include "../stencil_ct.cuh"
#include "../lbm.cuh"

// Jet initialization
__device__ void init_density_jet(pop_t __restrict__ *f,
                                 real_t __restrict__ *rho,
                                 const label_t x, const label_t y, const label_t z)
{
    const label_t id = idx(x, y, z);

    rho[id] = rho0;

    const real_t vx = real_t(0.0);
    const real_t vy = real_t(0.0);
    const real_t vz = real_t(0.0);

    constexpr_for<0, Q>([&] __device__(auto I)
                        {
        constexpr label_t i = decltype(I)::value;

        const real_t fi = feq<i>(rho[id], vx, vy, vz);
        f[fidx(id, i)] = save_pop(fi); });
}

// Inlet-plane initialization
__device__ void jet_mask(pop_t __restrict__ *f,
                         real_t __restrict__ *rho,
                         const label_t x, const label_t z)
{
    const label_t yB = 0;
    const label_t id = idx(x, yB, z);

    rho[id] = rho0;

    const real_t vx = real_t(0.0);
    const real_t vy = real_t(0.0);
    const real_t vz = real_t(0.0);

    constexpr_for<0, Q>([&] __device__(auto I)
                        {
        constexpr label_t i = decltype(I)::value;

        const real_t fi = feq<i>(rho[id], vx, vy, vz);
        f[fidx(id, i)] = save_pop(fi); });
}

#endif