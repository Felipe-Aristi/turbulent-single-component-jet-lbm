#ifndef LBM_CUH
#define LBM_CUH

#include "constants.cuh"
#include "stencil_ct.cuh"
#include "utilities/bounds.cuh"
#include "utilities/indexing.cuh"
#include "utilities/types.cuh"
#include "utilities/mathUtilities.cuh"
#include "utilities/constexprFor.cuh"

//----------- Distribution functions -----------

template <label_t I>
__device__ __forceinline__ real_t feq(const real_t rho,
                                      const real_t ux,
                                      const real_t uy,
                                      const real_t uz) noexcept
{
    constexpr real_t cx = static_cast<real_t>(D3Q27::cx<I>());
    constexpr real_t cy = static_cast<real_t>(D3Q27::cy<I>());
    constexpr real_t cz = static_cast<real_t>(D3Q27::cz<I>());
    constexpr real_t wi = D3Q27::w<I>();

    const real_t cu = ux * cx + uy * cy + uz * cz;
    const real_t usq = ux * ux + uy * uy + uz * uz;

    const real_t cu2 = cu * cu;

    const real_t A2eq = (cu * inv_cs2) - (usq * inv_2cs2) + (cu * cu) * inv_2cs4;
    const real_t A3eq = cu * (cu2 * inv_6cs6 - usq * inv_2cs4);

    return wi * rho * (A2eq + A3eq) + wi * (rho - static_cast<real_t>(1.0));
}

template <label_t I>
__device__ __forceinline__ real_t fneqr(const real_t Pixx,
                                        const real_t Pixy,
                                        const real_t Piyy,
                                        const real_t Piyz,
                                        const real_t Pizz,
                                        const real_t Pixz,
                                        const real_t ux,
                                        const real_t uy,
                                        const real_t uz) noexcept
{
    constexpr real_t Hxx = D3Q27::Hxx<I>();
    constexpr real_t Hxy = D3Q27::Hxy<I>();
    constexpr real_t Hyy = D3Q27::Hyy<I>();
    constexpr real_t Hyz = D3Q27::Hyz<I>();
    constexpr real_t Hzz = D3Q27::Hzz<I>();
    constexpr real_t Hxz = D3Q27::Hxz<I>();

    constexpr real_t Hxxy = D3Q27::Hxxy<I>();
    constexpr real_t Hxxz = D3Q27::Hxxz<I>();
    constexpr real_t Hxyy = D3Q27::Hxyy<I>();
    constexpr real_t Hxzz = D3Q27::Hxzz<I>();
    constexpr real_t Hyyz = D3Q27::Hyyz<I>();
    constexpr real_t Hyzz = D3Q27::Hyzz<I>();
    constexpr real_t Hxyz = D3Q27::Hxyz<I>();

    constexpr real_t wi = D3Q27::w<I>();

    const real_t A2neq = (Pixx * Hxx + real_t(2.0) * Pixy * Hxy + Piyy * Hyy + real_t(2.0) * Piyz * Hyz + Pizz * Hzz + real_t(2.0) * Pixz * Hxz) * inv_2cs4;

    const real_t a3xxy = Pixx * uy + real_t(2.0) * Pixy * ux;
    const real_t a3xxz = Pixx * uz + real_t(2.0) * Pixz * ux;
    const real_t a3xyy = Piyy * ux + real_t(2.0) * Pixy * uy;
    const real_t a3xzz = Pizz * ux + real_t(2.0) * Pixz * uz;
    const real_t a3yyz = Piyy * uz + real_t(2.0) * Piyz * uy;
    const real_t a3yzz = Pizz * uy + real_t(2.0) * Piyz * uz;
    const real_t a3xyz = Pixy * uz + Pixz * uy + Piyz * ux;

    const real_t A3neq = (a3xxy * Hxxy + a3xxz * Hxxz + a3xyy * Hxyy + a3xzz * Hxzz + a3yyz * Hyyz + a3yzz * Hyzz + real_t(2.0) * a3xyz * Hxyz) * inv_2cs6;

    return wi * (A2neq + A3neq);
}

//----------- Macroscopic fields calculation -----------

__device__ __forceinline__ void Mfields_calculation(const pop_t __restrict__ *f, real_t __restrict__ *rho,
                                                    real_t __restrict__ *ux, real_t __restrict__ *uy, real_t __restrict__ *uz,
                                                    real_t __restrict__ *Pixx, real_t __restrict__ *Pixy, real_t __restrict__ *Piyy,
                                                    real_t __restrict__ *Piyz, real_t __restrict__ *Pizz, real_t __restrict__ *Pixz,
                                                    const label_t x, const label_t y, const label_t z)
{

    const label_t id = idx(x, y, z);

    real_t sum = static_cast<real_t>(0.0);

    real_t jx = static_cast<real_t>(0.0);
    real_t jy = static_cast<real_t>(0.0);
    real_t jz = static_cast<real_t>(0.0);

    real_t Axx = static_cast<real_t>(0.0);
    real_t Axy = static_cast<real_t>(0.0);
    real_t Ayy = static_cast<real_t>(0.0);
    real_t Ayz = static_cast<real_t>(0.0);
    real_t Azz = static_cast<real_t>(0.0);
    real_t Axz = static_cast<real_t>(0.0);

    constexpr_for<0, Q>(
        [&] __device__(auto I)
        {
            constexpr label_t i = decltype(I)::value;

            const real_t fi = load_pop(f[fidx(id, i)]);

            sum += fi;

            constexpr real_t cx = static_cast<real_t>(D3Q27::cx<I>());
            constexpr real_t cy = static_cast<real_t>(D3Q27::cy<I>());
            constexpr real_t cz = static_cast<real_t>(D3Q27::cz<I>());

            jx += fi * cx;
            jy += fi * cy;
            jz += fi * cz;

            constexpr real_t Hxx = D3Q27::Hxx<i>();
            constexpr real_t Hxy = D3Q27::Hxy<i>();
            constexpr real_t Hyy = D3Q27::Hyy<i>();
            constexpr real_t Hyz = D3Q27::Hyz<i>();
            constexpr real_t Hzz = D3Q27::Hzz<i>();
            constexpr real_t Hxz = D3Q27::Hxz<i>();

            Axx += fi * Hxx;
            Axy += fi * Hxy;
            Ayy += fi * Hyy;
            Ayz += fi * Hyz;
            Azz += fi * Hzz;
            Axz += fi * Hxz;
        });

    const real_t rho_t = sum + real_t(1.0);
    const real_t rho_inv = static_cast<real_t>(1.0) / rho_t;

    rho[id] = rho_t;

    const real_t vx = (jx)*rho_inv;
    const real_t vy = (jy)*rho_inv;
    const real_t vz = (jz)*rho_inv;

    ux[id] = vx;
    uy[id] = vy;
    uz[id] = vz;

    Pixx[id] = Axx - rho_t * vx * vx;
    Pixy[id] = Axy - rho_t * vx * vy;
    Piyy[id] = Ayy - rho_t * vy * vy;
    Piyz[id] = Ayz - rho_t * vy * vz;
    Pizz[id] = Azz - rho_t * vz * vz;
    Pixz[id] = Axz - rho_t * vx * vz;
}

//----------- Streaming and collision -----------

__device__ __forceinline__ void ColliStream_calculation(pop_t __restrict__ *f, const real_t __restrict__ *rho,
                                                        const real_t __restrict__ *ux, const real_t __restrict__ *uy, const real_t __restrict__ *uz,
                                                        const real_t __restrict__ *Pixx, const real_t __restrict__ *Pixy, const real_t __restrict__ *Piyy,
                                                        const real_t __restrict__ *Piyz, const real_t __restrict__ *Pizz, const real_t __restrict__ *Pixz,
                                                        const label_t x, const label_t y, const label_t z)
{
    const label_t id = idx(x, y, z);

    const real_t rhol = rho[id];
    const real_t vx = ux[id];
    const real_t vy = uy[id];
    const real_t vz = uz[id];
    const real_t pixx = Pixx[id];
    const real_t pixy = Pixy[id];
    const real_t piyy = Piyy[id];
    const real_t piyz = Piyz[id];
    const real_t pizz = Pizz[id];
    const real_t pixz = Pixz[id];

    const real_t omega_eff = omega_sponge(y);
    const real_t oms = static_cast<real_t>(1.0) - omega_eff;

    constexpr_for<0, Q>(
        [&] __device__(auto I)
        {
            constexpr label_t i = decltype(I)::value;

            const real_t fieq = feq<i>(rhol, vx, vy, vz);
            const real_t fineqr = fneqr<i>(pixx, pixy, piyy, piyz, pizz, pixz, vx, vy, vz);

            const real_t fi = fieq + oms * fineqr;

            // const int xn = static_cast<int>(x) + D3Q27::cx<i>();
            // const int zn = static_cast<int>(z) + D3Q27::cz<i>();

            // // periodic boundary condition
            const int xn = wrapx(static_cast<int>(x) + D3Q27::cx<i>());
            const int zn = wrapz(static_cast<int>(z) + D3Q27::cz<i>());

            const int yn = static_cast<int>(y) + D3Q27::cy<i>();

            const label_t idn = idx(static_cast<label_t>(xn),
                                    static_cast<label_t>(yn),
                                    static_cast<label_t>(zn));

            f[fidx(idn, i)] = save_pop(fi);
        });
}

#endif
