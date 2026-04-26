#ifndef LAUNCH_CUH
#define LAUNCH_CUH

#include <cuda_runtime.h>

#include "utilities/cudaUtilities.cuh"
#include "utilities/cudaConfig.cuh"

#include "memory.cuh"
#include "kernels.cuh"

inline void launch_jetDensity(const CudaConfig &cfg, const LbmDevice &d, cudaStream_t stream = 0)
{
    jetDensity<<<cfg.grid, cfg.block, 0, stream>>>(d.f, d.rho);
    CUDA_CHECK(cudaGetLastError());
}

inline void launch_Injet(const CudaConfig &cfg, const LbmDevice &d, cudaStream_t stream = 0)
{
    Injet<<<grid2D_xz(cfg), block2D_xz(cfg), 0, stream>>>(d.f, d.rho);
    CUDA_CHECK(cudaGetLastError());
}

inline void launch_Macros(const CudaConfig &cfg, const LbmDevice &d, cudaStream_t stream = 0)
{
    Mfields<<<cfg.grid, cfg.block, 0, stream>>>(
        d.f, d.rho,
        d.ux, d.uy, d.uz,
        d.Pixx, d.Pixy, d.Piyy, d.Piyz, d.Pizz, d.Pixz);

    CUDA_CHECK(cudaGetLastError());
}

inline void launch_compute_total_tke(const CudaConfig &cfg,
                                     const LbmDevice &d,
                                     real_t *d_tke_total,
                                     cudaStream_t stream = 0)
{
    compute_total_tke<<<cfg.grid, cfg.block, 0, stream>>>(
        d.ux, d.uy, d.uz,
        d_tke_total);

    CUDA_CHECK(cudaGetLastError());
}

inline void launch_update_tke_average(real_t *d_tke_avg,
                                      const real_t *d_tke_total,
                                      unsigned int step,
                                      unsigned int init_step,
                                      cudaStream_t stream = 0)
{
    update_tke_average<<<1, 1, 0, stream>>>(
        d_tke_avg, d_tke_total, step, init_step);

    CUDA_CHECK(cudaGetLastError());
}

inline void launch_update_uy_average(const CudaConfig &cfg,
                                     const LbmDevice &d,
                                     real_t *d_uy_avg,
                                     unsigned int step,
                                     unsigned int step_uy_avg_start,
                                     cudaStream_t stream = 0)
{
    update_uy_average<<<cfg.grid, cfg.block, 0, stream>>>(
        d.uy, d_uy_avg, step, step_uy_avg_start);

    CUDA_CHECK(cudaGetLastError());
}

inline void launch_accumulate_radial_moments(const CudaConfig &cfg,
                                             const LbmDevice &d,
                                             const MeanFieldsDevice &mf,
                                             cudaStream_t stream = 0)
{
    accumulate_radial_moments<<<cfg.grid, cfg.block, 0, stream>>>(
        d.ux, d.uy, d.uz,
        mf.sum_uy, mf.sum_uy2,
        mf.sum_ur, mf.sum_ur2,
        mf.sum_uruy,
        mf.count);

    CUDA_CHECK(cudaGetLastError());
}

inline void launch_collistream(const CudaConfig &cfg, const LbmDevice &d, cudaStream_t stream = 0)
{
    ColliStream<<<cfg.grid, cfg.block, 0, stream>>>(
        d.f, d.rho,
        d.ux, d.uy, d.uz,
        d.Pixx, d.Pixy, d.Piyy, d.Piyz, d.Pizz, d.Pixz);

    CUDA_CHECK(cudaGetLastError());
}

inline void launch_inlet_bc(const CudaConfig &cfg, const LbmDevice &d, cudaStream_t stream = 0)
{
    inlet<<<grid2D_xz(cfg), block2D_xz(cfg), 0, stream>>>(
        d.f, d.rho,
        d.Pixx, d.Pixy, d.Piyy, d.Piyz, d.Pizz, d.Pixz);

    CUDA_CHECK(cudaGetLastError());
}

inline void launch_neumann_bc(const CudaConfig &cfg, const LbmDevice &d, cudaStream_t stream = 0)
{
    neumann<<<grid2D_xz(cfg), block2D_xz(cfg), 0, stream>>>(
        d.f, d.rho,
        d.ux, d.uy, d.uz,
        d.Pixx, d.Pixy, d.Piyy, d.Piyz, d.Pizz, d.Pixz);

    CUDA_CHECK(cudaGetLastError());
}

#endif
