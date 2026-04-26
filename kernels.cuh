#ifndef KERNELS_CUH
#define KERNELS_CUH

#include "constants.cuh"
#include "utilities/types.cuh"
#include "meanF/memoryMeanFields.cuh"

__global__ void jetDensity(pop_t __restrict__ *f, real_t __restrict__ *rho);

__global__ void Injet(pop_t __restrict__ *f, real_t __restrict__ *rho);

__global__ void Mfields(const pop_t __restrict__ *f, real_t __restrict__ *rho,
                        real_t __restrict__ *ux, real_t __restrict__ *uy, real_t __restrict__ *uz,
                        real_t __restrict__ *Pixx, real_t __restrict__ *Pixy, real_t __restrict__ *Piyy,
                        real_t __restrict__ *Piyz, real_t __restrict__ *Pizz, real_t __restrict__ *Pixz);

__global__ void ColliStream(pop_t __restrict__ *f, const real_t __restrict__ *rho,
                            const real_t __restrict__ *ux, const real_t __restrict__ *uy, const real_t __restrict__ *uz,
                            const real_t __restrict__ *Pixx, const real_t __restrict__ *Pixy, const real_t __restrict__ *Piyy,
                            const real_t __restrict__ *Piyz, const real_t __restrict__ *Pizz, const real_t __restrict__ *Pixz);

__global__ void inlet(pop_t __restrict__ *f, real_t __restrict__ *rho,
                      const real_t __restrict__ *Pixx, const real_t __restrict__ *Pixy, const real_t __restrict__ *Piyy,
                      const real_t __restrict__ *Piyz, const real_t __restrict__ *Pizz, const real_t __restrict__ *Pixz);

__global__ void neumann(pop_t __restrict__ *f, real_t __restrict__ *rho,
                        real_t __restrict__ *ux, real_t __restrict__ *uy, real_t __restrict__ *uz,
                        const real_t __restrict__ *Pixx, const real_t __restrict__ *Pixy, const real_t __restrict__ *Piyy,
                        const real_t __restrict__ *Piyz, const real_t __restrict__ *Pizz, const real_t __restrict__ *Pixz);

__global__ void compute_total_tke(const real_t *__restrict__ ux,
                                  const real_t *__restrict__ uy,
                                  const real_t *__restrict__ uz,
                                  real_t *__restrict__ tke_total);

__global__ void update_tke_average(real_t *tke_avg,
                                   const real_t *tke_total,
                                   unsigned int step,
                                   unsigned int init_step);

__global__ void update_uy_average(const real_t *__restrict__ uy,
                                  real_t *__restrict__ uy_avg,
                                  unsigned int step,
                                  unsigned int step_uy_avg_start);

__global__ void accumulate_radial_moments(const real_t *__restrict__ ux,
                                          const real_t *__restrict__ uy,
                                          const real_t *__restrict__ uz,
                                          profile_stat_t *__restrict__ sum_uy,
                                          profile_stat_t *__restrict__ sum_uy2,
                                          profile_stat_t *__restrict__ sum_ur,
                                          profile_stat_t *__restrict__ sum_ur2,
                                          profile_stat_t *__restrict__ sum_uruy,
                                          profile_count_t *__restrict__ count);

#endif
