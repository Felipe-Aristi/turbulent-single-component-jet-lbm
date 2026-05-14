#ifndef MEMORY_MEAN_FIELDS_CUH
#define MEMORY_MEAN_FIELDS_CUH

#include <cstdlib>
#include <cstdio>

#include "../constants.cuh"

#include "../utilities/cudaUtilities.cuh"
#include "../utilities/types.cuh"

using profile_stat_t = double;
using profile_count_t = unsigned long long;

struct MeanFieldsDevice
{
    real_t *tke_total = nullptr;
    real_t *tke_avg = nullptr;

    profile_stat_t *sum_uy = nullptr;
    profile_stat_t *sum_uy2 = nullptr;
    profile_stat_t *sum_ur = nullptr;
    profile_stat_t *sum_ur2 = nullptr;
    profile_stat_t *sum_uruy = nullptr;
    profile_count_t *count = nullptr;
};

struct MeanFieldsHost
{
    real_t tke_total = static_cast<real_t>(0);
    real_t tke_avg = static_cast<real_t>(0);
    real_t tke_avg_prev = static_cast<real_t>(0);
    real_t tke_total_prev = static_cast<real_t>(0);
    real_t delta = static_cast<real_t>(0);
    real_t abs_delta = static_cast<real_t>(0);

    profile_stat_t *sum_uy = nullptr;
    profile_stat_t *sum_uy2 = nullptr;
    profile_stat_t *sum_ur = nullptr;
    profile_stat_t *sum_ur2 = nullptr;
    profile_stat_t *sum_uruy = nullptr;
    profile_count_t *count = nullptr;
};

struct MeanFieldsState
{
    bool start_uy_average = false;
    bool first_tke_sample = true;
    unsigned int step_uy_avg_start = 0;
    unsigned int stable_energy_samples = 0;
    profile_count_t radial_sample_count = 0;
    static constexpr real_t tke_tolerance = static_cast<real_t>(1e-1);
};

inline constexpr std::size_t profileStatBytes = NradialProfileCells * sizeof(profile_stat_t);
inline constexpr std::size_t profileCountBytes = NradialProfileCells * sizeof(profile_count_t);

inline profile_stat_t *allocate_profile_stat_host(const char *name)
{
    profile_stat_t *ptr = static_cast<profile_stat_t *>(std::malloc(profileStatBytes));

    if (ptr == nullptr)
    {
        std::fprintf(stderr, "Host allocation failed for radial profile field: %s\n", name);
        std::exit(EXIT_FAILURE);
    }

    return ptr;
}

inline profile_count_t *allocate_profile_count_host()
{
    profile_count_t *ptr = static_cast<profile_count_t *>(std::malloc(profileCountBytes));

    if (ptr == nullptr)
    {
        std::fprintf(stderr, "Host allocation failed for radial profile count\n");
        std::exit(EXIT_FAILURE);
    }

    return ptr;
}

inline MeanFieldsDevice allocate_mean_fields_device()
{
    MeanFieldsDevice d{};

    CUDA_CHECK(cudaMalloc((void **)&d.tke_avg, sizeof(real_t)));
    CUDA_CHECK(cudaMalloc((void **)&d.tke_total, sizeof(real_t)));

    CUDA_CHECK(cudaMalloc((void **)&d.sum_uy, profileStatBytes));
    CUDA_CHECK(cudaMalloc((void **)&d.sum_uy2, profileStatBytes));
    CUDA_CHECK(cudaMalloc((void **)&d.sum_ur, profileStatBytes));
    CUDA_CHECK(cudaMalloc((void **)&d.sum_ur2, profileStatBytes));
    CUDA_CHECK(cudaMalloc((void **)&d.sum_uruy, profileStatBytes));
    CUDA_CHECK(cudaMalloc((void **)&d.count, profileCountBytes));

    CUDA_CHECK(cudaMemset(d.tke_total, 0, sizeof(real_t)));
    CUDA_CHECK(cudaMemset(d.tke_avg, 0, sizeof(real_t)));

    CUDA_CHECK(cudaMemset(d.sum_uy, 0, profileStatBytes));
    CUDA_CHECK(cudaMemset(d.sum_uy2, 0, profileStatBytes));
    CUDA_CHECK(cudaMemset(d.sum_ur, 0, profileStatBytes));
    CUDA_CHECK(cudaMemset(d.sum_ur2, 0, profileStatBytes));
    CUDA_CHECK(cudaMemset(d.sum_uruy, 0, profileStatBytes));
    CUDA_CHECK(cudaMemset(d.count, 0, profileCountBytes));

    return d;
}

inline MeanFieldsHost allocate_mean_fields_host()
{
    MeanFieldsHost h{};

    h.sum_uy = allocate_profile_stat_host("sum_uy");
    h.sum_uy2 = allocate_profile_stat_host("sum_uy2");
    h.sum_ur = allocate_profile_stat_host("sum_ur");
    h.sum_ur2 = allocate_profile_stat_host("sum_ur2");
    h.sum_uruy = allocate_profile_stat_host("sum_uruy");
    h.count = allocate_profile_count_host();

    return h;
}

inline void free_mean_fields_device(MeanFieldsDevice &d)
{
    CUDA_CHECK(cudaFree(d.tke_total));
    CUDA_CHECK(cudaFree(d.tke_avg));

    CUDA_CHECK(cudaFree(d.sum_uy));
    CUDA_CHECK(cudaFree(d.sum_uy2));
    CUDA_CHECK(cudaFree(d.sum_ur));
    CUDA_CHECK(cudaFree(d.sum_ur2));
    CUDA_CHECK(cudaFree(d.sum_uruy));
    CUDA_CHECK(cudaFree(d.count));

    d = MeanFieldsDevice{};
}

inline void free_mean_fields_host(MeanFieldsHost &h)
{
    std::free(h.sum_uy);
    std::free(h.sum_uy2);
    std::free(h.sum_ur);
    std::free(h.sum_ur2);
    std::free(h.sum_uruy);
    std::free(h.count);
    h = MeanFieldsHost{};
}

#endif
