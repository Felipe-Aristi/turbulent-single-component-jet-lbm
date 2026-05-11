#ifndef STENCIL_CT_CUH
#define STENCIL_CT_CUH

#include "utilities/types.cuh"
#include "constants.cuh"

namespace D3Q27
{
    inline constexpr int cx_ct[Q] = {
        0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1};

    inline constexpr int cy_ct[Q] = {
        0, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1};

    inline constexpr int cz_ct[Q] = {
        0, 0, 0, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1};

    template <label_t I>
    __host__ __device__ constexpr int cx() noexcept
    {
        return cx_ct[I];
    }

    template <label_t I>
    __host__ __device__ constexpr int cy() noexcept
    {
        return cy_ct[I];
    }

    template <label_t I>
    __host__ __device__ constexpr int cz() noexcept
    {
        return cz_ct[I];
    }

    template <label_t I>
    __host__ __device__ constexpr int s() noexcept
    {
        return cx<I>() * cx<I>() + cy<I>() * cy<I>() + cz<I>() * cz<I>();
    }

    template <label_t I>
    __host__ __device__ constexpr real_t w() noexcept
    {
        if constexpr (s<I>() == 0)
        {
            return real_t(8.0 / 27.0);
        }

        else if constexpr (s<I>() == 1)
        {
            return real_t(2.0 / 27.0);
        }

        else if constexpr (s<I>() == 2)
        {
            return real_t(1.0 / 54.0);
        }

        else
        {
            return real_t(1.0 / 216.0);
        }
    }

    template <label_t I>
    __host__ __device__ constexpr real_t B() noexcept
    {
        if constexpr (I == 0)
        {
            return real_t(-10.0 / 27.0);
        }

        else
        {
            return w<I>();
        }
    }

    // ===================================================
    // Second order Hermite polynomials
    // ===================================================

    template <label_t I>
    __host__ __device__ constexpr real_t Hxx() noexcept
    {
        return real_t(cx<I>() * cx<I>()) - cs2;
    }

    template <label_t I>
    __host__ __device__ constexpr real_t Hxy() noexcept
    {
        return real_t(cx<I>() * cy<I>());
    }

    template <label_t I>
    __host__ __device__ constexpr real_t Hyy() noexcept
    {
        return real_t(cy<I>() * cy<I>()) - cs2;
    }

    template <label_t I>
    __host__ __device__ constexpr real_t Hyz() noexcept
    {
        return real_t(cy<I>() * cz<I>());
    }

    template <label_t I>
    __host__ __device__ constexpr real_t Hzz() noexcept
    {
        return real_t(cz<I>() * cz<I>()) - cs2;
    }

    template <label_t I>
    __host__ __device__ constexpr real_t Hxz() noexcept
    {
        return real_t(cx<I>() * cz<I>());
    }

    // ===================================================
    // Third order Hermite polynomials
    // ===================================================
    template <label_t I>
    __host__ __device__ constexpr real_t Hxxy() noexcept
    {
        return real_t(cx<I>() * cx<I>() * cy<I>()) - cs2 * real_t(cy<I>());
    }

    template <label_t I>
    __host__ __device__ constexpr real_t Hxxz() noexcept
    {
        return real_t(cx<I>() * cx<I>() * cz<I>()) - cs2 * real_t(cz<I>());
    }

    template <label_t I>
    __host__ __device__ constexpr real_t Hxyy() noexcept
    {
        return real_t(cx<I>() * cy<I>() * cy<I>()) - cs2 * real_t(cx<I>());
    }

    template <label_t I>
    __host__ __device__ constexpr real_t Hxzz() noexcept
    {
        return real_t(cx<I>() * cz<I>() * cz<I>()) - cs2 * real_t(cx<I>());
    }

    template <label_t I>
    __host__ __device__ constexpr real_t Hyyz() noexcept
    {
        return real_t(cy<I>() * cy<I>() * cz<I>()) - cs2 * real_t(cz<I>());
    }

    template <label_t I>
    __host__ __device__ constexpr real_t Hyzz() noexcept
    {
        return real_t(cy<I>() * cz<I>() * cz<I>()) - cs2 * real_t(cy<I>());
    }

    template <label_t I>
    __host__ __device__ constexpr real_t Hxyz() noexcept
    {
        return real_t(cx<I>() * cy<I>() * cz<I>());
    }

    // ===============================================================================

    template <label_t I>
    __host__ __device__ constexpr real_t invcnorm() noexcept
    {
        if constexpr (s<I>() == 0)
        {
            return real_t(0);
        }

        else if constexpr (s<I>() == 1)
        {
            return real_t(1);
        }

        else if constexpr (s<I>() == 2)
        {
            return real_t(0.7071067811865475);
        }

        else
        {
            return real_t(0.5773502691896258);
        }
    }

    // Neighbours jump
    template <label_t I>
    __host__ __device__ constexpr int offset() noexcept
    {
        return cx<I>() + static_cast<int>(NX) * cy<I>() +
               static_cast<int>(NX) * static_cast<int>(NY) * cz<I>();
    }

    template <label_t I>
    __host__ __device__ constexpr int offset_xz() noexcept
    {
        return cx<I>() + static_cast<int>(NX) * static_cast<int>(NY) * cz<I>();
    }

}

#endif